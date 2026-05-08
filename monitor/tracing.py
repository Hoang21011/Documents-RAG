import os
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()

# ─── Kiểm tra cấu hình môi trường ────────────────────────────────────────────
# Langfuse v3+ tự động đọc từ biến môi trường:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
_pk   = (os.getenv("LANGFUSE_PUBLIC_KEY", "") or "").strip('"\'')
_sk   = (os.getenv("LANGFUSE_SECRET_KEY", "") or "").strip('"\'')
_host = (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").strip('"\'')

if _pk and _sk:
    print(f"[Tracing] Langfuse đã sẵn sàng → {_host}")
else:
    print("[Tracing] Thiếu LANGFUSE_PUBLIC_KEY hoặc LANGFUSE_SECRET_KEY trong .env")


# ─── Decorator helpers (5 observation types) ─────────────────────────────────
#
#  ┌─────────────┬──────────────────────────────────────────────────────────┐
#  │ as_type     │ Dùng khi nào                                             │
#  ├─────────────┼──────────────────────────────────────────────────────────┤
#  │ span        │ Bước xử lý logic chung (rerank, save history, ...)       │
#  │ generation  │ Gọi LLM → hiện token usage, model, cost trên dashboard   │
#  │ retriever   │ Tìm kiếm tài liệu từ vector DB (Milvus hybrid search)    │
#  │ embedding   │ Tạo vector embedding cho query hoặc document             │
#  │ chain       │ Nhóm nhiều step thành sub-pipeline, dùng cho root trace  │
#  └─────────────┴──────────────────────────────────────────────────────────┘
#
# Cách dùng trong orchestrator:
#
#   client = get_client()
#   with client.start_as_current_span(name="step", as_type="span") as span:
#       span.update(input="...", output="...", metadata={...})
#
# Hoặc dùng decorator @observe cho root function:
#
#   @observe(as_type="chain")
#   def ask_stream(self, ...):
#       ...

def trace_span(name: str):
    """
    [SPAN] Bước xử lý logic chung không phải LLM call hay DB call đặc thù.
    Ví dụ: query_rewriting (logic), mmr_reranking, save_history.
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
    [CHAIN] Root trace bao toàn bộ pipeline RAG.
    Gom nhóm tất cả sub-steps thành một chain duy nhất trên dashboard.
    Ví dụ: rag_pipeline (ask_stream).
    """
    return observe(name=name, as_type="chain")


# ─── Flush ────────────────────────────────────────────────────────────────────

def flush():
    """Đẩy toàn bộ trace còn pending lên Langfuse server trước khi app tắt."""
    try:
        get_client().flush()
    except Exception as e:
        print(f"[Tracing] Lỗi khi flush: {e}")
