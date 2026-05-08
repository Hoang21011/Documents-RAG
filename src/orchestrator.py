from src.retrieval import Retrieval
from src.rerank_and_format_chunks import RerankerAndFormatter
from src.generation import Generation
from database.connection import DatabaseManager
from monitor.logging import get_logger
from monitor.tracing import flush as langfuse_flush
from langfuse import observe, get_client
from datetime import datetime
from typing import List, Dict, Generator
import json
import time
import os


class Orchestrator:
    def __init__(self):
        print("[Orchestrator] Khởi tạo các module (Retrieval, Reranker, Generation)...")
        self.retriever = Retrieval()
        self.formatter = RerankerAndFormatter()
        self.generator = Generation()

        # Monitoring
        self.logger = get_logger("orchestrator")

        # MongoDB lưu lịch sử chat
        self.db_manager = DatabaseManager()
        self.mongo_db = self.db_manager.get_mongo_db()
        self.logs_collection = self.mongo_db['chat_logs']

        # Log latency ra file riêng
        self.latency_log_path = "/Users/nghia/Documents/khoa_luan_tot_nghiep/logs/latency.log"

    RELEVANCE_WARNING_THRESHOLD = 0.5

    # ─── Helpers ──────────────────────────────────────────────────────────────

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
            self.logger.error(f"Failed to write latency log: {e}")

    # ─── Step 0: Query Rewriting ──────────────────────────────────────────────

    def _generate_standalone_query(self, query: str, chat_history: List[str]) -> str:
        """
        [SPAN] Dùng LLM viết lại câu hỏi thành standalone query (De-contextualization).
        """
        client = get_client()
        with client.start_as_current_observation(
            name="query_rewriting",
            as_type="span",
            input={"query": query, "history_len": len(chat_history)}
        ) as span:
            if not chat_history:
                span.update(output=query, metadata={"rewritten": False})
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
                standalone = self.generator.generate_raw(prompt)
                result = standalone.strip() if standalone else query
            except Exception as e:
                self.logger.warning(f"[Orchestrator] Lỗi khi viết lại query: {e}")
                result = f"{' '.join(chat_history[-1:])} {query}"

            span.update(output=result, metadata={"rewritten": result != query})
            return result

    # ─── Step 1: Retrieval ────────────────────────────────────────────────────

    def _retrieve(self, standalone_query: str, filter_dict: dict,
                  top_k: int = 5, alpha: float = 0.4) -> List[Dict]:
        """
        [RETRIEVER] Hybrid search trên Milvus (Dense + Sparse).
        """
        client = get_client()
        with client.start_as_current_observation(
            name="milvus_hybrid_search",
            as_type="retriever",
            input={"query": standalone_query, "top_k": top_k, "alpha": alpha, "filter": filter_dict}
        ) as span:
            results = []
            for attempt in range(2):
                try:
                    results = self.retriever.search(
                        standalone_query,
                        chat_history=None,
                        filter_dict=filter_dict,
                        top_k=top_k,
                        alpha=alpha
                    )
                    break
                except Exception as e:
                    err_msg = str(e)
                    if any(k in err_msg for k in ["too_many_pings", "GOAWAY", "Fail connecting"]):
                        self.logger.warning(f"Milvus connection issue (Attempt {attempt+1}): {e}. Re-initializing...")
                        try:
                            from pymilvus import MilvusClient
                            self.retriever.client = MilvusClient(uri=self.retriever.db_uri)
                        except Exception:
                            pass
                        time.sleep(1.5)
                    else:
                        raise

            span.update(
                output={"num_chunks": len(results)},
                metadata={"top_k": top_k, "alpha": alpha}
            )
            return results

    # ─── Step 2: Reranking ────────────────────────────────────────────────────

    def _rerank(self, query: str, results: List[Dict], lambda_mult: float = 0.5) -> List[Dict]:
        """
        [SPAN] MMR Rerank + Lost-in-the-Middle reorder.
        """
        client = get_client()
        with client.start_as_current_observation(
            name="mmr_reranking",
            as_type="span",
            input={"query": query, "num_candidates": len(results), "lambda_mult": lambda_mult}
        ) as span:
            reranked  = self.formatter.mmr_rerank(query, results, lambda_mult)
            reordered = self.formatter.lost_in_the_middle_reorder(reranked)
            span.update(output={"num_reranked": len(reordered)})
            return reordered

    # ─── Step 3: Generation ───────────────────────────────────────────────────

    def _generate(self, query: str, session_id: str):
        """
        [GENERATION] Stream tokens từ LLM (Ollama / Qwen2.5-7B).
        Yields từng token ra ngoài để SSE stream.
        """
        client = get_client()
        with client.start_as_current_observation(
            name="llm_generation",
            as_type="generation",
            input=query,
            model="qwen2.5-7b"
        ) as span:
            start = time.time()
            full_answer = ""
            tokens_out  = 0

            for token in self.generator.generate_stream(query, session_id):
                full_answer += token
                tokens_out  += 1
                yield token

            generation_time = round(time.time() - start, 3)
            span.update(
                output=full_answer,
                usage_details={"output": tokens_out},
                metadata={"latency_ms": int(generation_time * 1000)}
            )

    # ─── Step 4: Save History ─────────────────────────────────────────────────

    def _save_history(self, session_id: str, query: str, full_answer: str,
                      contexts: List[str], filter_dict: dict, metrics: Dict):
        """
        [SPAN] Lưu lịch sử vào Redis + MongoDB.
        """
        client = get_client()
        with client.start_as_current_observation(
            name="save_history",
            as_type="span",
            input={"session_id": session_id, "query": query}
        ) as span:
            try:
                redis_client = self.db_manager.get_redis()
                history_key  = f"session:{session_id}:history"
                redis_client.rpush(history_key, f"User: {query}")
                redis_client.rpush(history_key, f"AI: {full_answer}")
                redis_client.expire(history_key, 86400)

                self.logs_collection.insert_one({
                    "session_id": session_id,
                    "query":      query,
                    "answer":     full_answer,
                    "contexts":   contexts,
                    "timestamp":  datetime.now(),
                    "filters":    filter_dict,
                    "latency":    metrics
                })
                span.update(output={"status": "ok"})
            except Exception as e:
                self.logger.error(f"Save history failed: {e}")
                span.update(
                    output={"status": "error", "error": str(e)},
                    level="ERROR"
                )

    # ─── Root trace: ask_stream ───────────────────────────────────────────────

    @observe(as_type="chain", name="rag_pipeline")
    def ask_stream(self, query: str, session_id: str, filter_dict: dict = None) -> Generator:
        """
        [CHAIN root] Generator cho SSE streaming.
        @observe(as_type='chain') tạo root trace bao toàn bộ pipeline.
        Các sub-steps dùng start_as_current_observation để tạo child spans lồng bên trong.

        Observation type mapping:
            ask_stream                → chain      (root)
            _generate_standalone_query → span
            _retrieve                 → retriever
            _rerank                   → span
            _generate                 → generation
            _save_history             → span
        """
        try:
            start_total = time.time()
            self.logger.info(f"New query received: {query}", extra={"model": "qwen2.5-7b"})

            # Lấy chat history từ Redis
            redis_client = self.db_manager.get_redis()
            history_key  = f"session:{session_id}:history"
            chat_history = []
            try:
                chat_history = [x.decode("utf-8") for x in redis_client.lrange(history_key, -4, -1)]
            except Exception:
                pass

            # ── Step 0: Query Rewriting ─────────────────────────────────────
            yield json.dumps({"type": "step", "message": "🧠 Đang xử lý ngữ cảnh câu hỏi..."})
            start_qr = time.time()
            standalone_query = self._generate_standalone_query(query, chat_history)
            qr_time = round(time.time() - start_qr, 3)
            self.logger.info(f"Standalone Query: {standalone_query}")

            # ── Step 1: Retrieval ────────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "🔍 Đang tìm kiếm tài liệu liên quan..."})
            start_ret = time.time()
            results = self._retrieve(standalone_query, filter_dict)
            retrieval_time = round(time.time() - start_ret, 3)
            contexts_for_ragas = [chunk["content"] for chunk in results]

            # ── Step 2: Reranking ────────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "📊 Đang xếp hạng và lọc thông tin..."})
            start_rerank = time.time()
            reordered = self._rerank(query, results)
            rerank_time = round(time.time() - start_rerank, 3)

            enriched_chunks = self._enrich_chunks(reordered)
            yield json.dumps({"type": "sources", "chunks": enriched_chunks})

            markdown_context = self.formatter.format_to_markdown(reordered)
            self.generator.cache_context(session_id, markdown_context)

            # ── Step 3: Generation ───────────────────────────────────────────
            yield json.dumps({"type": "step", "message": "🤖 Đang tổng hợp câu trả lời..."})
            start_gen = time.time()
            full_answer = ""
            for token in self._generate(query, session_id):
                full_answer += token
                yield json.dumps({"type": "token", "content": token})
            generation_time = round(time.time() - start_gen, 3)

            total_time = round(time.time() - start_total, 3)
            metrics = {
                "query_rewrite_sec": qr_time,
                "retrieval_sec":     retrieval_time,
                "rerank_sec":        rerank_time,
                "generation_sec":    generation_time,
                "total_sec":         total_time
            }
            self._log_latency(session_id, metrics)

            # ── Step 4: Save History ─────────────────────────────────────────
            self._save_history(session_id, query, full_answer,
                               contexts_for_ragas, filter_dict, metrics)

            self.logger.info(
                f"Pipeline finished in {total_time}s",
                extra={"latency_ms": int(total_time * 1000), "model": "qwen2.5-7b"}
            )

            # ── Done event ───────────────────────────────────────────────────
            yield json.dumps({
                "type":   "done",
                "answer": full_answer,
                "chunks": enriched_chunks,
            })

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)})
        finally:
            langfuse_flush()

    def ask(self, query: str, session_id: str, filter_dict: dict = None) -> str:
        full_answer = ""
        for event_str in self.ask_stream(query, session_id, filter_dict):
            try:
                event = json.loads(event_str)
                if event["type"] == "done":
                    full_answer = event["answer"]
            except Exception:
                pass
        return full_answer
