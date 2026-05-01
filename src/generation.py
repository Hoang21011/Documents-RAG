import os
import redis
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Generation:
    _instance = None
    _llm = None
    _redis_client = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Generation, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        base_dir_env = os.getenv("BASE_DIR", "/Users/nghia/Documents/khoa_luan_tot_nghiep")
        self.base_dir = Path(base_dir_env)
        
        # 1. Initialize DatabaseManager (Redis)
        from database.connection import DatabaseManager
        self.db_manager = DatabaseManager()
        self.redis = self.db_manager.get_redis()
            
        # 2. Load System Prompt
        prompt_path = self.base_dir / "system_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
            
        # 3. Load GGUF Model (Singleton)
        if Generation._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("Please install llama-cpp-python first.")
                
            model_path_env = os.getenv("LLM_MODEL_PATH", "LLM/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
            if model_path_env.startswith("./"):
                model_path = self.base_dir / model_path_env[2:]
            elif not model_path_env.startswith("/"):
                model_path = self.base_dir / model_path_env
            else:
                model_path = Path(model_path_env)
                
            print(f"Loading GGUF model from {model_path}...")
            
            if not model_path.exists():
                print(f"[CẢNH BÁO] Không tìm thấy file GGUF tại {model_path}. "
                      "Hệ thống chạy ở chế độ không có LLM (chỉ retrieval).")
                Generation._llm = None
            else:
                import multiprocessing
                # Tự động lấy số core vật lý (thường là 8 trên M1/M2)
                threads = max(multiprocessing.cpu_count() // 2, 4)
                
                Generation._llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=-1, # Metal on Mac
                    n_ctx=4096,
                    n_threads=threads,
                    flash_attn=True, # Tăng tốc độ xử lý context
                    verbose=False
                )
        self.llm = Generation._llm
        
        self.failure_threshold = 3
        self._initialized = True


    def cache_context(self, session_id: str, context: str):
        """Helper function to mock/store context into Redis before Generation"""
        self.redis.set(f"context:{session_id}", context, ex=3600) # Expire in 1 hour
        
    def generate(self, query: str, session_id: str = "default") -> str:
        # 1. Fetch Context
        context = self.redis.get(f"context:{session_id}")
        if not context:
            context = "" # Không in lỗi, để trống cho prompt tự trả lời "Không đủ dữ liệu."
            
        # 2. Check Failure Threshold
        failure_key = f"failures:{session_id}"
        failures = int(self.redis.get(failure_key) or 0)
        
        if failures >= self.failure_threshold:
            return "Tôi không thể tìm thấy thông tin bạn cần. Vui lòng liên hệ bộ phận hỗ trợ hoặc chuyên viên con người qua email support@example.com hoặc hotline 1900-xxxx để được trợ giúp."
            
        # 3. Prepare Prompt for Qwen2.5
        if self.llm is None:
            return "[Hệ thống đang chạy không có LLM] Model GGUF chưa được tải."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{query}"}
        ]
        
        print("Generating response via Qwen2.5-7B-Instruct (GGUF)...")
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9
        )
        
        answer = response['choices'][0]['message']['content'].strip()
        
        # 4. Handle Strict Fallback Logic
        if "Không đủ dữ liệu" in answer:
            self.redis.incr(failure_key)
            self.redis.expire(failure_key, 3600)
            return "Không đủ dữ liệu."
        else:
            self.redis.set(failure_key, 0)
            
        return answer

    def generate_raw(self, prompt: str) -> str:
        """Sinh văn bản trực tiếp không qua cache context, dùng cho rewrite query."""
        if self.llm is None: return ""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Generation] Lỗi sinh văn bản raw: {e}")
            return ""

    def generate_stream(self, query: str, session_id: str = "default"):
        """
        Generator yielding real tokens from llama.cpp (stream=True).
        Also returns full accumulated answer at the end via StopIteration value.
        """
        context = self.redis.get(f"context:{session_id}") or ""
        
        failure_key = f"failures:{session_id}"
        failures = int(self.redis.get(failure_key) or 0)
        if failures >= self.failure_threshold:
            fallback = ("Tôi không thể tìm thấy thông tin bạn cần. "
                        "Vui lòng liên hệ bộ phận hỗ trợ qua email hoặc hotline 1900-xxxx.")
            yield fallback
            return

        if self.llm is None:
            yield "[LLM chưa được tải] Các tài liệu liên quan đã hiển thị ở phần References."
            return

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{query}"}
        ]
        
        full_answer = ""
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            stream=True          # ← Real token streaming
        )
        
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_answer += token
                yield token          # Emit each token as it arrives

        # Fallback tracking after stream completes
        if "Không đủ dữ liệu" in full_answer:
            self.redis.incr(failure_key)
            self.redis.expire(failure_key, 3600)
        else:
            self.redis.set(failure_key, 0)


if __name__ == "__main__":
    from src.rerank_and_format_chunks import RerankerAndFormatter
    from src.retrieval import Retrieval
    
    # Fake session
    session_id = "test_session_123"
    
    print("\n--- 1. RETRIEVING & RERANKING ---")
    retriever = Retrieval()
    formatter = RerankerAndFormatter()
    
    query = "Quy định về đạo văn và liêm chính học thuật"
    filters = {"pham_vi": "NEU/Kinh tế quốc dân"}
    results = retriever.search(query, filter_dict=filters, top_k=5, alpha=0.5)
    markdown_context = formatter.process(query, results, lambda_mult=0.5)
    
    print("\n--- 2. INITIALIZING GENERATION (GGUF) ---")
    generator = Generation()
    
    # Cache Context to Redis
    generator.cache_context(session_id, markdown_context)
    
    print("\n--- 3. GENERATING ANSWER ---")
    answer = generator.generate(query, session_id)
    
    print("\n=== TRẢ LỜI ===")
    print(answer)
    
    print("\n--- 4. TESTING OUT-OF-CONTEXT (FALLBACK) ---")
    query_out = "Tại sao bầu trời màu xanh?"
    # Tạo context trống hoặc không liên quan
    generator.cache_context("test_session_456", "Không có thông tin liên quan.")
    
    for i in range(4): # Chạy 4 lần để kích hoạt fallback
        ans = generator.generate(query_out, "test_session_456")
        print(f"Lần {i+1}: {ans}")
