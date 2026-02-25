from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from schema import RagQueryRequest, RenameThreadRequest
from rag_query_pipeline import rag_query_stream, chatbot, checkpointer
from utils_threads import retrieve_all_threads, delete_thread, rename_thread

from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(
    title = "RAG Backend",
    description = "FastAPI backend for RTU syllabus RAG system",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

@app.get("/threads")
def get_threads():
    return {"threads": retrieve_all_threads(checkpointer)}

@app.get("/threads/{thread_id}")
def get_thread_messages(thread_id: str):
    messages = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values.get("messages", [])
    formatted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue

        formatted.append({
            "role": role,
            "content": msg.content
        })
    
    return {"messages": formatted}

@app.delete("/threads/{thread_id}")
def delete_thread_endpoint(thread_id: str):
    return delete_thread(checkpointer, thread_id)

@app.patch("/threads/{thread_id}/rename")
def rename_thread_endpoint(thread_id: str, request: RenameThreadRequest):
    return rename_thread(chatbot, thread_id, request.title)

@app.post("/rag/query")
async def query_rag(request: RagQueryRequest):
    return StreamingResponse(
        rag_query_stream(request),
        media_type = "text/plain"
    )