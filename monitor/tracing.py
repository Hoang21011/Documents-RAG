import os
from dotenv import load_dotenv

load_dotenv()

class RAGTracer:
    """
    Wrapper cho Langfuse SDK v4+ để ghi trace pipeline RAG.
    
    Thư mục liên quan:
      - monitor/tracing.py  ← file này (định nghĩa)
      - src/orchestrator.py ← nơi gọi create_trace() / flush()
      - Langfuse Dashboard  ← nơi xem kết quả (LANGFUSE_BASE_URL trong .env)
    """
    
    def __init__(self, env: str = "dev", tier: str = "free"):
        self.env = env
        self.tier = tier
        self._langfuse = None
        self._current_trace_ctx = None
        
        # Chỉ khởi tạo nếu có đủ config
        has_config = (
            os.getenv("LANGFUSE_PUBLIC_KEY") and
            os.getenv("LANGFUSE_SECRET_KEY")
        )
        if has_config:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse()
            except Exception as e:
                print(f"[Tracing] Không thể khởi tạo Langfuse: {e}")

    def create_trace(self, name: str, feature: str, model: str,
                     session_id: str = None, user_id: str = None):
        """
        Tạo trace mới bằng Langfuse v4 API (start_observation).
        Trả về context manager để dùng với `with` hoặc tự động flush.
        """
        if self._langfuse is None:
            return None
        
        try:
            ctx = self._langfuse.start_observation(
                name=name,
                type="trace",
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "env": self.env,
                    "feature": feature,
                    "tier": self.tier,
                    "model": model,
                },
                tags=[self.env, feature, self.tier, model],
            )
            self._current_trace_ctx = ctx
            return ctx
        except Exception as e:
            print(f"[Tracing] Lỗi khi tạo trace: {e}")
            return None

    def flush(self):
        """Đẩy toàn bộ trace còn lại lên Langfuse server."""
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception as e:
                print(f"[Tracing] Lỗi khi flush: {e}")
