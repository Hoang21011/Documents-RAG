import os
from dotenv import load_dotenv
from typing import Optional
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

# ─── Khởi tạo Langfuse client singleton ──────────────────────────────────────
# @observe sẽ tự động dùng client này thông qua langfuse_context
_pk   = (os.getenv("LANGFUSE_PUBLIC_KEY", "") or "").strip('"\'')
_sk   = (os.getenv("LANGFUSE_SECRET_KEY", "") or "").strip('"\'')
_host = (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").strip('"\'')

langfuse = None
if _pk and _sk:
    try:
        langfuse = Langfuse(public_key=_pk, secret_key=_sk, host=_host)
        print(f"[Tracing] Langfuse đã sẵn sàng → {_host}")
    except Exception as e:
        print(f"[Tracing] Không thể khởi tạo Langfuse: {e}")
else:
    print("[Tracing] Thiếu LANGFUSE_PUBLIC_KEY hoặc LANGFUSE_SECRET_KEY trong .env")


# ─── Decorator helpers (5 observation types) ────────────────────────────────
#
#  Langfuse hỗ trợ 5 loại observation type. Dùng đúng type giúp dashboard
#  phân loại, lọc và tính chi phí chính xác hơn.
#
#  ┌─────────────┬──────────────────────────────────────────────────────────┐
#  │ as_type     │ Dùng khi nào                                             │
#  ├─────────────┼──────────────────────────────────────────────────────────┤
#  │ span        │ Bước xử lý logic chung (rerank, save history, ...)       │
#  │ generation  │ Gọi LLM → hiện token usage, model, cost trên dashboard   │
#  │ retriever   │ Tìm kiếm tài liệu từ vector DB (Milvus hybrid search)    │
#  │ embedding   │ Tạo vector embedding cho query hoặc document             │
#  │ chain       │ Nhóm nhiều step thành sub-pipeline (retrieve→rerank)     │
#  └─────────────┴──────────────────────────────────────────────────────────┘

def trace_span(name: str):
    """
    [SPAN] Bước xử lý logic chung không phải LLM call hay DB call đặc thù.
    Ví dụ: query_rewriting (logic), mmr_reranking, lost_in_middle_reorder,
           save_history, post-processing.
    """
    return observe(name=name, as_type="span")


def trace_generation(name: str):
    """
    [GENERATION] Bước gọi LLM để sinh text.
    Langfuse sẽ hiển thị: model name, input/output tokens, latency, cost.
    Ví dụ: llm_generation (câu trả lời cuối), generate_raw (query rewrite).
    """
    return observe(name=name, as_type="generation")


def trace_retriever(name: str):
    """
    [RETRIEVER] Bước truy xuất tài liệu từ vector database.
    Langfuse sẽ hiển thị: documents trả về, query, số lượng kết quả.
    Ví dụ: milvus_hybrid_search, dense_search, sparse_search.
    """
    return observe(name=name, as_type="retriever")


def trace_embedding(name: str):
    """
    [EMBEDDING] Bước tạo vector embedding cho query hoặc document.
    Langfuse sẽ hiển thị: model embedding, input text, vector dimension.
    Ví dụ: encode_query_dense, encode_query_sparse (BM25).
    """
    return observe(name=name, as_type="embedding")


def trace_chain(name: str):
    """
    [CHAIN] Nhóm nhiều step liên quan thành một sub-pipeline có thể tái sử dụng.
    Dùng để gom nhóm các bước liên quan lại, giúp trace dễ đọc hơn trên dashboard.
    Ví dụ: retrieval_chain (embedding → search → rerank), rag_pipeline (root).
    """
    return observe(name=name, as_type="chain")


# ─── Utility để cập nhật metadata trong lúc chạy ─────────────────────────────

def update_current_observation(**kwargs):
    """
    Gọi bên trong một hàm đã được @observe để cập nhật metadata/input/output
    của observation hiện tại trong runtime.

    Ví dụ:
        update_current_observation(
            output="...",
            metadata={"latency_ms": 120},
            usage={"input": 150, "output": 300}
        )
    """
    try:
        langfuse_context.update_current_observation(**kwargs)
    except Exception:
        pass  # Không làm crash pipeline nếu Langfuse lỗi


def update_current_trace(**kwargs):
    """
    Cập nhật metadata của root trace (session_id, user_id, tags, metadata).

    Ví dụ:
        update_current_trace(
            session_id="abc-123",
            user_id="user-456",
            tags=["prod", "qwen2.5-7b", "QA"],
            metadata={"env": "prod", "tier": "free"}
        )
    """
    try:
        langfuse_context.update_current_trace(**kwargs)
    except Exception:
        pass


def flush():
    """Đẩy toàn bộ trace còn pending lên Langfuse server trước khi app tắt."""
    if langfuse:
        try:
            langfuse.flush()
        except Exception as e:
            print(f"[Tracing] Lỗi khi flush: {e}")
