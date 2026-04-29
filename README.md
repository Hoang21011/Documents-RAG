# 🎓 NEU Connect – Hệ thống Trợ lý Tra cứu Quy chế AI

NEU Connect là một hệ thống **Legal RAG** (Retrieval-Augmented Generation) hiện đại, được thiết kế đặc biệt để hỗ trợ sinh viên và cán bộ tra cứu các quy định, quy chế nội bộ của Trường Đại học Kinh tế Quốc dân (NEU).

Hệ thống sử dụng mô hình ngôn ngữ lớn (LLM) chạy cục bộ, kết hợp với tìm kiếm lai (Hybrid Search) để đảm bảo câu trả lời chính xác, bảo mật và có trích dẫn nguồn đầy đủ.

---

## 🏗️ Kiến trúc hệ thống
Xem chi tiết tại: [architecture.md](./architecture.md)

---

## 🛠️ Công nghệ sử dụng
*   **AI/LLM**: Qwen2.5-7B-Instruct (GGUF), llama-cpp-python.
*   **Search**: Hybrid Search (Dense SBERT + Sparse BM25), MMR Reranking.
*   **Backend**: FastAPI (Python), Server-Sent Events (SSE) cho Streaming.
*   **Frontend**: React (Vite), Modern UI NEU Theme.
*   **Database**: Milvus Lite (Vector), Redis (Session), MongoDB (Logs).
*   **Monitoring**: Langfuse (Tracing), JSON Logging.

---

## 🚀 Hướng dẫn cài đặt (Local Development)

### 1. Chuẩn bị môi trường
*   Python 3.11+ và Node.js 18+.
*   Cài đặt Redis và MongoDB trên máy (hoặc chạy qua Docker).

### 2. Cài đặt Backend
```bash
# Tại thư mục gốc dự án
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Cài llama-cpp-python hỗ trợ GPU (Mac M1/M2/M3)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### 3. Cấu hình biến môi trường
Tạo file `.env` từ mẫu sau:
```env
# Database
MILVUS_DB_URI=./data/vector_database/milvus.db
REDIS_HOST=localhost
REDIS_PORT=6379
MONGO_URI=mongodb://localhost:27017

# Model paths
LLM_MODEL_PATH=LLM/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
EMBEDDING_MODEL_NAME=keepitreal/vietnamese-sbert
BM25_MODEL_PATH=data/vector_database/bm25_model.json

# Monitoring (tùy chọn)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

### 4. Tải Model LLM
Hệ thống sử dụng model Qwen2.5 7B. Bạn cần tải file GGUF về thư mục `LLM/`:
```bash
mkdir -p LLM
# Tải qua huggingface-cli hoặc script python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', local_dir='LLM'); hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf', local_dir='LLM')"
```

### 5. Khởi chạy hệ thống

**Chạy Backend (Terminal 1):**
```bash
PYTHONPATH=. python main.py --api
```

**Chạy Frontend (Terminal 2):**
```bash
cd front_end
npm install
npm run dev
```
Truy cập: `http://localhost:5173`

---

## 🐳 Triển khai với Docker

Hệ thống hỗ trợ chạy bằng Docker để đồng bộ môi trường:

```bash
# Build image
docker build -t neu-connect-rag .

# Chạy container (cần mount các thư mục data và model)
docker run -p 8000:8000 \
  -v $(pwd)/LLM:/app/LLM \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  neu-connect-rag
```

---

## 🧪 Đánh giá (Evaluation)
Hệ thống tích hợp sẵn pipeline đánh giá tự động bằng **RAGAS** và **Gemini 2.5 Flash** (đóng vai trò LLM Judge).
1. Cập nhật `GEMINI_API_KEY` trong `.env`.
2. Chạy script đánh giá:
```bash
PYTHONPATH=. python eval/evalutation.py
```
Kết quả sẽ được lưu vào file `eval/eval_result.txt` và MongoDB để phân tích lỗi (failure analysis).

---

## 🤝 CI/CD
Dự án sử dụng GitHub Actions:
*   **CI**: Tự động kiểm tra lỗi code và build frontend mỗi khi push.
*   **CD**: Tự động build và push Docker image lên Docker Hub khi merge vào branch `main`.
