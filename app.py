from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from schema import RagQueryRequest
from rag_query_pipeline import rag_query_stream

app = FastAPI(
    title = "RAG Backend",
    description = "FastAPI backend for RTU syllabus RAG system",
    version = "0.1.0"
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

@app.post("/rag/query")
def query_rag(request: RagQueryRequest):
    return StreamingResponse(
        rag_query_stream(request),
        media_type = "text/event-stream"
    )