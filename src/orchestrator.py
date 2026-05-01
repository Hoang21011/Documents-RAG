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
import os

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
        
        # Thư mục logs cho latency
        self.latency_log_path = "/Users/nghia/Documents/khoa_luan_tot_nghiep/logs/latency.log"

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

    def _log_latency(self, session_id: str, metrics: Dict[str, float]):
        """Ghi log latency vào file riêng biệt."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "metrics": metrics
            }
            os.makedirs(os.path.dirname(self.latency_log_path), exist_ok=True)
            with open(self.latency_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to write latency log: {e}")

    def _generate_standalone_query(self, query: str, chat_history: List[str]) -> str:
        """
        Sử dụng LLM để viết lại câu hỏi dựa trên lịch sử hội thoại (De-contextualization).
        Giúp giải quyết triệt để vấn đề Context Drift.
        """
        if not chat_history:
            return query
            
        history_text = "\n".join(chat_history)
        prompt = f"""Dựa vào lịch sử hội thoại dưới đây và câu hỏi mới nhất, hãy viết lại câu hỏi mới nhất thành một câu hỏi độc lập (standalone query) chứa đầy đủ ngữ cảnh để có thể dùng tìm kiếm trong tài liệu quy chế.
Nếu câu hỏi đã đủ rõ ràng, hãy giữ nguyên.
CHỈ TRẢ VỀ CÂU HỎI ĐÃ VIẾT LẠI, KHÔNG GIẢI THÍCH GÌ THÊM.

LỊCH SỬ:
{history_text}

CÂU HỎI MỚI: {query}
CÂU HỎI ĐỘC LẬP:"""

        try:
            # Dùng method generate_raw (giả định có trong Generation) hoặc tạo tạm thời
            standalone = self.generator.generate_raw(prompt)
            return standalone.strip() if standalone else query
        except Exception as e:
            print(f"[Orchestrator] Lỗi khi viết lại query: {e}")
            return f"{' '.join(chat_history[-1:])} {query}" # Fallback simple join

    def ask_stream(self, query: str, session_id: str, filter_dict: dict = None) -> Generator:
        """
        Generator cho SSE streaming với đo lường latency chi tiết.
        """
        try:
            start_total = time.time()
            self.logger.info(f"New query received: {query}", extra={"model": "qwen2.5-7b"})

            # Lấy chat history từ Redis
            redis_client = self.db_manager.get_redis()
            history_key = f"session:{session_id}:history"
            chat_history = []
            try:
                chat_history = [x.decode('utf-8') for x in redis_client.lrange(history_key, -4, -1)]
            except:
                pass

            # ── Step 0: Query Rewriting (Tránh Context Drift) ──────────────
            yield json.dumps({"type": "step", "message": "🧠 Đang xử lý ngữ cảnh câu hỏi..."})
            standalone_query = self._generate_standalone_query(query, chat_history)
            self.logger.info(f"Standalone Query: {standalone_query}")

            # Trace
            try:
                self.tracer.create_trace(name="RAG Query", feature="QA", model="qwen2.5-7b", session_id=session_id)
            except Exception as e:
                self.logger.warning(f"Tracing error: {e}")

            # ── Step 1: Retrieval ─────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "🔍 Đang tìm kiếm tài liệu liên quan..."})
            start_ret = time.time()
            
            top_k = 5
            alpha = 0.4
            lambda_mult = 0.5

            # Dùng standalone_query để retrieval chính xác hơn
            results = []
            for attempt in range(2):
                try:
                    # Lưu ý: retriever.search giờ nhận query đã qua xử lý lịch sử
                    results = self.retriever.search(standalone_query, chat_history=None, filter_dict=filter_dict, top_k=top_k, alpha=alpha)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "too_many_pings" in err_msg or "GOAWAY" in err_msg or "Fail connecting" in err_msg:
                        self.logger.warning(f"Milvus connection issue (Attempt {attempt+1}): {e}. Re-initializing client...")
                        try:
                            from pymilvus import MilvusClient
                            self.retriever.client = MilvusClient(uri=self.retriever.db_uri)
                        except: pass
                        time.sleep(1.5)
                    else:
                        raise e
            
            retrieval_time = round(time.time() - start_ret, 3)
            contexts_for_ragas = [chunk["content"] for chunk in results]

            # ── Step 2: MMR Rerank ────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "📊 Đang xếp hạng và lọc thông tin..."})
            start_rerank = time.time()

            # Rerank dựa trên query gốc để giữ tính phù hợp cao nhất với mong muốn cuối cùng của user
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

            # ── Step 4: Latency Logging ────────────────────────────────────
            metrics = {
                "retrieval_sec": retrieval_time,
                "rerank_sec": rerank_time,
                "generation_sec": generation_time,
                "total_sec": total_time
            }
            self._log_latency(session_id, metrics)

            # ── Step 5: Lưu Redis + MongoDB ────────────────────────────────
            try:
                redis_client.rpush(history_key, f"User: {query}")
                redis_client.rpush(history_key, f"AI: {full_answer}")
                redis_client.expire(history_key, 86400)

                self.logs_collection.insert_one({
                    "session_id": session_id,
                    "query": query,
                    "answer": full_answer,
                    "contexts": contexts_for_ragas,
                    "timestamp": datetime.now(),
                    "filters": filter_dict,
                    "latency": metrics
                })
                self.logger.info(f"Pipeline finished in {total_time}s")
            except Exception as e:
                self.logger.error(f"Save history failed: {e}")

            # ── Done event ────────────────────────────────────────────────
            yield json.dumps({
                "type": "done",
                "answer": full_answer,
                "chunks": enriched_chunks,
            })

            if self.tracer:
                try: self.tracer.flush()
                except: pass

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)})

    def ask(self, query: str, session_id: str, filter_dict: dict = None) -> str:
        full_answer = ""
        for event_str in self.ask_stream(query, session_id, filter_dict):
            try:
                event = json.loads(event_str)
                if event["type"] == "done":
                    full_answer = event["answer"]
            except:
                pass
        return full_answer
