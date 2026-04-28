import asyncio
import json
import uuid
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

_orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo Orchestrator (và Milvus) đúng 1 lần khi server start."""
    global _orchestrator
    print("[API] Khởi tạo DatabaseManager...")
    from database.connection import DatabaseManager
    DatabaseManager().connect()

    print("[API] Khởi tạo Orchestrator (Milvus + Embedding + LLM)...")
    from src.orchestrator import Orchestrator
    _orchestrator = Orchestrator()
    print("[API] Sẵn sàng nhận request!")
    yield
    # Cleanup khi shutdown (nếu cần)
    print("[API] Shutdown.")

app = FastAPI(title="Legal RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Narrow down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_orchestrator():
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator chưa sẵn sàng.")
    return _orchestrator

# ── Models ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    filters: Optional[dict] = None

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE Streaming endpoint. Frontend connects with EventSource / fetch+ReadableStream.
    Each data line is a JSON object with `type` field:
      step | sources | token | done | error
    """
    session_id = req.session_id or str(uuid.uuid4())
    orch = get_orchestrator()

    def event_generator():
        for event_str in orch.ask_stream(req.query, session_id, req.filters):
            # SSE format: "data: <json>\n\n"
            yield f"data: {event_str}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Lấy lịch sử hội thoại từ Redis cho một session."""
    from database.connection import DatabaseManager
    redis = DatabaseManager().get_redis()
    history = redis.lrange(f"session:{session_id}:history", 0, -1)
    return {"session_id": session_id, "history": history}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
