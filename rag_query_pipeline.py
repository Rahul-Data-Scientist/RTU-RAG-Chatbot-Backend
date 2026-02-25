from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from schema import RagQueryRequest
from system_messages import final_prompt_system_message, rewrite_query_system_message

from typing import List
import os

load_dotenv()

def init_models():
    client = QdrantClient(url="http://qdrant:6333")

    vector_store = QdrantVectorStore(
        collection_name = "rtu_curriculum",
        embedding = OpenAIEmbeddings(model = "text-embedding-ada-002"),
        client = client
    )

    
    answer_llm = ChatOpenAI(model = "gpt-4.1-mini", streaming = True)
    rewrite_llm = ChatOpenAI(model = "gpt-4.1-mini", streaming = False)
    title_llm = ChatOpenAI(model = "gpt-4.1-mini", streaming = False)
    
    retriever = vector_store.as_retriever(
            search_kwargs = {
                "k": 6,
                "search_params": {
                    "hnsw_ef": 256
                }
                }
            )
    
    return client, retriever, answer_llm, rewrite_llm, title_llm

client, retriever, answer_llm, rewrite_llm, title_llm = init_models()

def build_qdrant_filter(semester: int, subject: str, unit: int):
    return models.Filter(
        must = [
            models.FieldCondition(
                key = "metadata.semester",
                match = models.MatchValue(value = semester)
            ),
            models.FieldCondition(
                key = "metadata.subject",
                match = models.MatchValue(value = subject)
            ),
            models.FieldCondition(
                key = "metadata.unit",
                match = models.MatchValue(value = unit)
            )
        ]
    )

DB_URI = os.environ["DB_URI"]

class ChatState(MessagesState):
    semester: int
    subject: str
    unit: int
    docs: List[Document]
    rewritten_query: str
    context: str
    conversation_title: str | None
    summary: str
    last_summarized_index: int

def rewrite_query(state: ChatState):
    messages = [rewrite_query_system_message]
    messages.extend(state['messages'][-4:])
    messages.append(HumanMessage(content = f"Rewrite this question: {state['messages'][-1].content}"))
    rewritten = rewrite_llm.invoke(messages)
    return {"rewritten_query": rewritten.content.strip()}

def generate_title(state: ChatState):
    if state.get("conversation_title") is not None:
        return {}
    messages = [
        SystemMessage(
            content="""
Generate a short atomic conversation title (max 5 words, try to keep 3-4 words only if possible).
No quotes. No punctuation.
Just the title.
"""
        ),
        HumanMessage(content=state['rewritten_query'])
    ]

    title = title_llm.invoke(messages).content.strip()
    title = " ".join(title.split()[:6])
    return {"conversation_title": title}


def retrieve(state: ChatState):
    qdrant_filter = build_qdrant_filter(state['semester'], state['subject'], state['unit'])
    docs = retriever.invoke(state.get("rewritten_query") or state['messages'][-1].content, filter = qdrant_filter)
    return {"docs": docs}

def build_context(state: ChatState):
    context = "\n\n".join([f"Source {i + 1}\n{d.page_content}" for i, d in enumerate(state['docs'])]).strip()
    return {"context": context}

def generate_answer(state: ChatState):
    if not state.get("context"):
        return {"messages": [AIMessage(content = "I couldn't find relevant syllabus information.")]}
    messages = [final_prompt_system_message]
    if state.get("summary"):
        messages.append(
            SystemMessage(
        content=f"""
    This is a summary of the prior conversation.
    Use it as background memory only.

    SUMMARY:
    {state['summary']}
    """
    )
        )
    
    # send only last 4 messages (including the latest user message) to the answer llm
    recent_messages = state["messages"][:-1][-3:]
    messages.extend(recent_messages)
    messages.append(
        HumanMessage(
            content=f"""
QUESTION:
{state['messages'][-1].content}

CONTEXT:
{state['context']}
"""
        )
    )

    response = answer_llm.invoke(messages)

    return {"messages": [response]}

def should_summarize(state: ChatState):
    if len(state["messages"]) - state.get("last_summarized_index", 0) >= 6:
        return "summarization needed"
    else:
        return "no summarization needed"

def summarize_conversation(state: ChatState):
    start = state.get("last_summarized_index", 0)
    new_messages = state["messages"][start:]

    if not new_messages:
        return {}

    existing_summary = state.get("summary")

    if existing_summary:
        instruction = (
            f"""
You are maintaining a running summary of a conversation.

Existing summary:
{existing_summary}

Update the summary using ONLY the new conversation messages provided.
Keep it concise but preserve important facts, decisions, and context.
"""
        )
    else:
        instruction = (
            """
Create a concise summary of the conversation.
Preserve key topics, user intent, and important context.
"""
        )

    message_for_summary = [
        SystemMessage(
            content="You update conversation summaries incrementally."
        ),
        *new_messages,
        HumanMessage(content=instruction),
    ]

    response = rewrite_llm.invoke(message_for_summary)

    return {
        "summary": response.content.strip(),
        "last_summarized_index": len(state["messages"]),
    }

graph = StateGraph(ChatState)

graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate_title", generate_title)
graph.add_node("retrieve_docs", retrieve)
graph.add_node("build_context", build_context)
graph.add_node("generate_answer", generate_answer)
graph.add_node("summarize", summarize_conversation)

graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "generate_title")
graph.add_edge("rewrite_query", "retrieve_docs")
graph.add_edge("retrieve_docs", "build_context")
graph.add_edge("build_context", "generate_answer")
graph.add_conditional_edges(
    "generate_answer",
    should_summarize,
    {
        "summarization needed": "summarize",
        "no summarization needed": END
    }
)
graph.add_edge("summarize", END)

# --- create checkpointer once ---
_checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = _checkpointer_cm.__enter__()

checkpointer.setup()

chatbot = graph.compile(checkpointer=checkpointer)

def rag_query_stream(request: RagQueryRequest):
        CONFIG = {"configurable": {"thread_id": request.thread_id}}
        input_state = {
            "messages": [HumanMessage(request.query)],
            "semester": request.semester,
            "subject": request.subject,
            "unit": request.unit
        }

        for message_chunk, metadata in chatbot.stream(
            input_state,
            config = CONFIG,
            stream_mode = "messages"
        ):
            if (metadata.get("langgraph_node") == "generate_answer" 
                and isinstance(message_chunk, AIMessage) 
                and message_chunk.content):
                yield message_chunk.content
