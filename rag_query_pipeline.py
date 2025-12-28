from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import tracing_v2_enabled

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from schema import RagQueryRequest, ChatMessage
from system_messages import final_prompt_system_message, rewrite_query_system_message

load_dotenv()

MAX_TURNS = 6

def init_models():
    client = QdrantClient(url="http://qdrant:6333")

    vector_store = QdrantVectorStore(
        collection_name = "rtu_curriculum",
        embedding = OpenAIEmbeddings(model = "text-embedding-ada-002"),
        client = client
    )

    
    answer_llm = ChatOpenAI(model = "gpt-4.1-nano", streaming = True)
    rewrite_llm = ChatOpenAI(model = "gpt-4.1-nano", streaming = False)
    
    retriever = vector_store.as_retriever(
            search_kwargs = {
                "k": 6,
                "search_params": {
                    "hnsw_ef": 256
                }
                }
            )
    
    return client, retriever, answer_llm, rewrite_llm

client, retriever, answer_llm, rewrite_llm = init_models()

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


@traceable(name = "filtered_retrieval")
def retrieve_similar_chunks(query, semester, subject, unit):
    qdrant_filter = build_qdrant_filter(semester, subject, unit)
    return retriever.invoke(query, filter = qdrant_filter)

@traceable(name = "query_rewrite")
def rewrite_query(query, chat_history: list[ChatMessage]):
    rewrite_messages = [rewrite_query_system_message]
    
    for msg in chat_history[-4:]:
        if msg.role == "user":
            rewrite_messages.append(HumanMessage(content = msg.content))
        else:
            rewrite_messages.append(AIMessage(content = msg.content))
    
    rewrite_messages.append(HumanMessage(content = f"Rewrite this question: {query}"))
    
    rewritten = rewrite_llm.invoke(rewrite_messages)
    return rewritten.content.strip()

@traceable(name = "context_construction")
def build_context(relevant_docs):
    return "\n\n".join([f"Source {i + 1}\n{d.page_content}" for i, d in enumerate(relevant_docs)])

def build_message(query, context, chat_history: list[ChatMessage]):
    messages = [final_prompt_system_message]
    
    for msg in chat_history[-MAX_TURNS:]:
        if msg.role == "user":
            messages.append(HumanMessage(content = msg.content))
        else:
            messages.append(AIMessage(content = msg.content))
    
    messages.append(HumanMessage(
        content=(
            "QUESTION:\n"
            f"{query}\n\n"
            "CONTEXT:\n"
            f"{context}"
        )
    ))
    
    return messages

@traceable(name = "rag_query", run_type = "chain")
def rag_query_stream(request: RagQueryRequest):
    
    with tracing_v2_enabled(project_name = None):
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.metadata.update({
                "semester": request.filters.semester,
                "subject": request.filters.subject,
                "unit": request.filters.unit
            })
        
        retrieval_query = rewrite_query(request.query, request.chat_history) or request.query
        
        docs = retrieve_similar_chunks(
            retrieval_query, 
            request.filters.semester, 
            request.filters.subject, 
            request.filters.unit
            )
        
        context = build_context(docs)
        
        if not context:
            yield "The provided context does not contain sufficient information to answer this question."
        
        final_prompt = build_message(request.query, context, request.chat_history)
        for chunk in answer_llm.stream(final_prompt):
            if chunk.content:
                yield chunk.content