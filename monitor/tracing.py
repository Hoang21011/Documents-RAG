import os
from dotenv import load_dotenv
from typing import Optional, Any, List
from contextlib import contextmanager
from langfuse import Langfuse, propagate_attributes

load_dotenv()

class RAGTracer:
    """
    Wrapper cho Langfuse SDK để ghi trace pipeline RAG.
    Sử dụng API hiện đại (Observation-centric).
    """
    
    def __init__(self, env: str = "dev", tier: str = "free"):
        self.env = env
        self.tier = tier
        self._langfuse = None
        
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        
        # Loại bỏ dấu ngoặc kép nếu có
        host = host.strip('"').strip("'")
        if pk: pk = pk.strip('"').strip("'")
        if sk: sk = sk.strip('"').strip("'")

        if pk and sk:
            try:
                self._langfuse = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=host
                )
                print(f"[Tracing] Đã khởi tạo Langfuse với host: {host}")
            except Exception as e:
                print(f"[Tracing] Không thể khởi tạo Langfuse: {e}")
        else:
            print("[Tracing] Thiếu cấu hình LANGFUSE_PUBLIC_KEY hoặc LANGFUSE_SECRET_KEY trong .env")

    @property
    def client(self) -> Optional[Langfuse]:
        """Trả về instance Langfuse client."""
        return self._langfuse

    @contextmanager
    def observe(self, name: str, session_id: str = None, user_id: str = None, 
                input: Any = None, metadata: dict = None, tags: List[str] = None):
        """
        Context manager để trace một operation.
        Sử dụng start_as_current_observation và propagate_attributes.
        """
        if self._langfuse is None:
            yield None
            return

        # Gộp metadata mặc định
        meta = {
            "env": self.env,
            "tier": self.tier,
            **(metadata or {})
        }

        try:
            # Khởi tạo observation hiện tại
            with self._langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input=input,
                metadata=meta
            ) as span:
                # Lan truyền các thuộc tính trace xuống các sub-observations
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id,
                    tags=tags or [],
                    metadata=meta
                ):
                    yield span
        except Exception as e:
            print(f"[Tracing] Lỗi trong quá trình observe: {e}")
            yield None

    def flush(self):
        """Đẩy toàn bộ trace còn lại lên Langfuse server."""
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception as e:
                print(f"[Tracing] Lỗi khi flush: {e}")
