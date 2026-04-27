import os
from langfuse import Langfuse

class RAGTracer:
    def __init__(self, env: str = "dev", tier: str = "free"):
        """
        Khởi tạo Tracer.
        Lưu ý: Langfuse SDK tự động đọc cấu hình kết nối từ biến môi trường:
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_HOST
        """
        self.langfuse = Langfuse()
        self.env = env
        self.tier = tier

    def create_trace(self, name: str, feature: str, model: str, session_id: str = None, user_id: str = None):
        """
        Khởi tạo một trace theo dõi các quy trình trong hệ thống RAG với các label chỉ định.
        """
        trace = self.langfuse.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            # Các tags giúp lọc và nhóm các trace trên Langfuse dashboard
            tags=[self.env, feature, self.tier, model],
            # Metadata chi tiết lưu các label được yêu cầu
            metadata={
                "env": self.env,
                "feature": feature,
                "tier": self.tier,
                "model": model
            }
        )
        return trace

    def flush(self):
        """
        Đảm bảo tất cả các dữ liệu trace còn lại được đẩy lên Langfuse server.
        Nên gọi hàm này trước khi ứng dụng kết thúc.
        """
        self.langfuse.flush()
