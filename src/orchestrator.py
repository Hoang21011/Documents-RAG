from src.retrieval import Retrieval
from src.rerank_and_format_chunks import RerankerAndFormatter
from src.generation import Generation
from database.connection import DatabaseManager
from monitor.logging import get_logger
from monitor.tracing import RAGTracer
from datetime import datetime
import json

class Orchestrator:
    def __init__(self):
        print("[Orchestrator] Khởi tạo các module (Retrieval, Reranker, Generation)...")
        self.retriever = Retrieval()
        self.formatter = RerankerAndFormatter()
        self.generator = Generation()
        
        # Khởi tạo Hệ thống Monitoring
        self.logger = get_logger("orchestrator")
        self.tracer = RAGTracer()
        
        # Sử dụng MongoDB để lưu lại lịch sử chat (Logging)
        self.db_manager = DatabaseManager()
        self.mongo_db = self.db_manager.get_mongo_db()
        self.logs_collection = self.mongo_db['chat_logs']
        
    def ask(self, query: str, session_id: str, filter_dict: dict = None) -> str:
        self.logger.info(f"New query received: {query}", extra={"model": "qwen2.5-7b"})
        
        # Khởi tạo trace cho session hiện tại
        try:
            trace = self.tracer.create_trace(name="RAG Query", feature="QA", model="qwen2.5-7b", session_id=session_id)
        except Exception as e:
            self.logger.warning(f"Không thể khởi tạo Langfuse tracer (có thể thiếu cấu hình): {e}")
            
        # 1. Hybrid Retrieval từ Milvus
        top_k = 5
        alpha = 0.4
        lambda_mult = 0.5
        
        self.logger.info("Executing retrieval...")
        results = self.retriever.search(query, filter_dict=filter_dict, top_k=top_k, alpha=alpha)
        
        # Lấy mảng chunks thô để phục vụ RAGAS Evaluation sau này
        contexts_for_ragas = [chunk['content'] for chunk in results]
        
        # 2. MMR Rerank & Markdown Formatting
        self.logger.info("Executing rerank and formatting...")
        markdown_context = self.formatter.process(query, results, lambda_mult=lambda_mult)
        
        # Lưu trữ context vào Redis cho session này
        self.generator.cache_context(session_id, markdown_context)
        
        # 3. LLM Generation
        self.logger.info("Executing LLM generation...")
        answer = self.generator.generate(query, session_id)
        
        # 4. Lưu lại lịch sử hội thoại (Session Store) vào Redis
        redis_client = self.db_manager.get_redis()
        history_key = f"session:{session_id}:history"
        redis_client.rpush(history_key, f"User: {query}")
        redis_client.rpush(history_key, f"AI: {answer}")
        # Hết hạn lịch sử sau 24 giờ
        redis_client.expire(history_key, 86400)
        
        # 5. Lưu lại lịch sử truy vấn vào MongoDB để RAGAS theo dõi sau này
        try:
            self.logs_collection.insert_one({
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "contexts": contexts_for_ragas, # Rất quan trọng cho RAGAS (faithfulness, context_precision)
                "timestamp": datetime.now(),
                "filters": filter_dict
            })
            self.logger.info("Saved query logs to MongoDB successfully")
        except Exception as e:
            self.logger.error(f"Failed to save log to MongoDB: {e}", exc_info=True)
            
        # 6. Push Tracking lên Langfuse
        self.logger.info("Flushing trace to Langfuse...")
        try:
            self.tracer.flush()
        except Exception as e:
            self.logger.error(f"Failed to flush trace: {e}")
            
        return answer
