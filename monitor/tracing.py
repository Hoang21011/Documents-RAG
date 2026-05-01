import os
from dotenv import load_dotenv

load_dotenv()

class RAGTracer:
    """
    Wrapper cho Langfuse SDK để ghi trace pipeline RAG.
    """
    
    def __init__(self, env: str = "dev", tier: str = "free"):
        self.env = env
        self.tier = tier
        self._langfuse = None
        
        has_config = (
            os.getenv("LANGFUSE_PUBLIC_KEY") and
            os.getenv("LANGFUSE_SECRET_KEY")
        )
        if has_config:
            try:
                from langfuse import Langfuse
                # Khởi tạo đối tượng Langfuse với các tham số tường minh
                self._langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
            except Exception as e:
                print(f"[Tracing] Không thể khởi tạo Langfuse: {e}")

    def create_trace(self, name: str, feature: str, model: str,
                     session_id: str = None, user_id: str = None):
        """
        Tạo trace mới. 
        Sử dụng cú pháp langfuse.trace(name=...) hoặc langfuse.span(...) theo gợi ý.
        """
        if self._langfuse is None:
            return None
        
        try:
            # Ưu tiên dùng .trace(name=...)
            if hasattr(self._langfuse, 'trace'):
                return self._langfuse.trace(
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                    metadata={
                        "env": self.env,
                        "feature": feature,
                        "model": model,
                    }
                )
            
            # Thử dùng .span(...)
            elif hasattr(self._langfuse, 'span'):
                return self._langfuse.span(
                    name=name, 
                    session_id=session_id
                )
            
            else:
                print(f"[Tracing] Langfuse object thiếu phương thức trace/span.")
                return None
                
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
