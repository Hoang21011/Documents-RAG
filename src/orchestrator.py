from src.retrieval import Retrieval
from src.rerank_and_format_chunks import RerankerAndFormatter
from src.generation import Generation
from database.connection import DatabaseManager
from monitor.logging import get_logger
from monitor.tracing import RAGTracer
from datetime import datetime
from typing import List, Dict, Any, Generator
import json
import time

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

    # ─── Ngưỡng cảnh báo độ liên quan thấp ───────────────────────────────
    RELEVANCE_WARNING_THRESHOLD = 0.5

    def _enrich_chunks(self, reranked_chunks: List[Dict]) -> List[Dict]:
        """Gắn index [1], [2], ... và nhãn cảnh báo low_relevance vào mỗi chunk."""
        enriched = []
        for i, chunk in enumerate(reranked_chunks, start=1):
            score = chunk.get("distance", 1.0)
            enriched.append({
                "id": i,
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata", {}),
                "score": round(score, 4),
                "low_relevance": score < self.RELEVANCE_WARNING_THRESHOLD,
            })
        return enriched

    def ask_stream(self, query: str, session_id: str, filter_dict: dict = None) -> Generator:
        """
        Generator cho SSE streaming. Trả về từng sự kiện dưới dạng JSON string.
        Các event type:
          - step:       thông báo bước đang chạy
          - sources:    danh sách chunks đã retrieve (để render UI citations ngay)
          - token:      từng token của câu trả lời (streaming)
          - done:       kết thúc, trả về answer đầy đủ và metadata
          - error:      thông báo lỗi
        """
        try:
            start_total = time.time()
            self.logger.info(f"New query received: {query}", extra={"model": "qwen2.5-7b"})

            # Trace
            try:
                self.tracer.create_trace(name="RAG Query", feature="QA", model="qwen2.5-7b", session_id=session_id)
            except Exception as e:
                self.logger.warning(f"Langfuse tracer unavailable: {e}")

            # ── Step 1: Retrieval ─────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "🔍 Đang tìm kiếm tài liệu liên quan..."})
            start_ret = time.time()
            
            top_k = 5
            alpha = 0.4
            lambda_mult = 0.5

            results = self.retriever.search(query, filter_dict=filter_dict, top_k=top_k, alpha=alpha)
            retrieval_time = round(time.time() - start_ret, 3)
            contexts_for_ragas = [chunk["content"] for chunk in results]

            # ── Step 2: MMR Rerank ────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "📊 Đang xếp hạng và lọc thông tin..."})
            start_rerank = time.time()

            reranked = self.formatter.mmr_rerank(query, results, lambda_mult)
            reordered = self.formatter.lost_in_the_middle_reorder(reranked)
            enriched_chunks = self._enrich_chunks(reordered)
            
            rerank_time = round(time.time() - start_rerank, 3)

            # Gửi sources ngay sau khi có để frontend render Citations panel
            yield json.dumps({"type": "sources", "chunks": enriched_chunks})

            # Tạo markdown context cho LLM
            markdown_context = self.formatter.format_to_markdown(reordered)
            self.generator.cache_context(session_id, markdown_context)

            # ── Step 3: Generation (streaming tokens) ────────────────────
            yield json.dumps({"type": "step", "message": "🤖 Đang tổng hợp câu trả lời..."})
            start_gen = time.time()

            full_answer = ""
            for token in self.generator.generate_stream(query, session_id):
                full_answer += token
                yield json.dumps({"type": "token", "content": token})
            
            generation_time = round(time.time() - start_gen, 3)
            total_time = round(time.time() - start_total, 3)

            # ── Step 4: Lưu Redis + MongoDB ────────────────────────────────
            redis_client = self.db_manager.get_redis()
            history_key = f"session:{session_id}:history"
            redis_client.rpush(history_key, f"User: {query}")
            redis_client.rpush(history_key, f"AI: {full_answer}")
            redis_client.expire(history_key, 86400)

            try:
                self.logs_collection.insert_one({
                    "session_id": session_id,
                    "query": query,
                    "answer": full_answer,
                    "contexts": contexts_for_ragas,
                    "timestamp": datetime.now(),
                    "filters": filter_dict,
                    "latency": {
                        "retrieval": retrieval_time,
                        "rerank": rerank_time,
                        "generation": generation_time,
                        "total": total_time
                    }
                })
                self.logger.info(f"Pipeline finished in {total_time}s (Ret: {retrieval_time}s, Rerank: {rerank_time}s, Gen: {generation_time}s)")
            except Exception as e:
                self.logger.error(f"MongoDB save failed: {e}", exc_info=True)

            # ── Done event ────────────────────────────────────────────────
            yield json.dumps({
                "type": "done",
                "answer": full_answer,
                "chunks": enriched_chunks,
            })

            try:
                self.tracer.flush()
            except Exception:
                pass

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)})

    # ── Phương thức đồng bộ (dùng cho CLI / testing) ──────────────────────
    def ask(self, query: str, session_id: str, filter_dict: dict = None) -> str:
        full_answer = ""
        for event_str in self.ask_stream(query, session_id, filter_dict):
            event = json.loads(event_str)
            if event["type"] == "done":
                full_answer = event["answer"]
        return full_answer
