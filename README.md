# Trợ lý Pháp luật AI – Legal RAG System

Hệ thống hỏi đáp văn bản pháp luật thông minh, sử dụng **Hybrid Search** (Dense + Sparse), **MMR Reranking**, **Qwen2.5-7B-Instruct (GGUF)**, và **React.js** với Citations tương tác.

---

## 📁 Cấu trúc thư mục

```
khoa_luan_tot_nghiep/
├── api/                  # FastAPI backend
│   └── main.py
├── database/             # Kết nối database
│   └── connection.py     # Redis + MongoDB Manager
├── eval/                 # Đánh giá hệ thống
│   ├── evalutation.py    # RAGAS + Custom metrics
│   └── golden_dataset.json
├── front_end/            # React (Vite) UI
│   └── src/
├── LLM/                  # GGUF model (không commit)
├── monitor/              # Logging & Tracing
│   ├── logging.py
│   └── tracing.py
├── src/                  # Pipeline chính
│   ├── chunking.py
│   ├── embed.py
│   ├── retrieval.py
│   ├── rerank_and_format_chunks.py
│   ├── generation.py
│   └── orchestrator.py
├── data/                 # Dữ liệu (không commit)
├── system_prompt.txt
├── main.py               # Điểm khởi động
├── requirements.txt
└── .env
```

---

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.11+
- **Node.js**: 18+
- **macOS** (Apple Silicon khuyến nghị để chạy LLM với Metal GPU)
- **RAM**: Tối thiểu 16 GB (model 7B chiếm ~5-6 GB)

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1 – Clone và cài đặt dependencies Python

```bash
git clone <repo-url>
cd khoa_luan_tot_nghiep

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# Cài thư viện Python
pip install -r requirements.txt

# Cài llama-cpp-python với Metal GPU (macOS Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### Bước 2 – Cài và khởi động Redis + MongoDB

```bash
# Cài Redis
brew install redis
brew services start redis

# Cài MongoDB
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Bước 3 – Cấu hình file `.env`

Tạo file `.env` ở thư mục gốc:

```env
BASE_DIR=/đường/dẫn/đến/khoa_luan_tot_nghiep

# Vector DB & Embedding
EMBEDDING_MODEL_NAME=keepitreal/vietnamese-sbert
MILVUS_DB_URI=./data/vector_database/milvus.db
MILVUS_COLLECTION_NAME=legal_docs_collection
BM25_MODEL_PATH=data/vector_database/bm25_model.json

# LLM
LLM_MODEL_PATH=LLM/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

# Database
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=legal_rag
REDIS_HOST=localhost
REDIS_PORT=6379

# Evaluation (tùy chọn)
GEMINI_API_KEY=your_key_here

# Monitoring (tùy chọn)
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

### Bước 4 – Tải model GGUF

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', local_dir='LLM')
hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf', local_dir='LLM')
"
```

### Bước 5 – Nhúng dữ liệu (chỉ cần chạy 1 lần)

```bash
# Đặt các file dữ liệu vào data/raw/
PYTHONPATH=. python src/embed.py
```

### Bước 6 – Cài đặt Frontend

```bash
cd front_end
npm install
cd ..
```

---

## ▶️ Khởi chạy ứng dụng

> **Lưu ý**: Luôn chạy từ **thư mục gốc** (`khoa_luan_tot_nghiep/`) với `PYTHONPATH=.`

### Chạy với giao diện Web (khuyến nghị)

**Terminal 1 – FastAPI Backend:**
```bash
PYTHONPATH=. python main.py --api
```
→ Chờ đến khi thấy: `[API] Sẵn sàng nhận request!`

**Terminal 2 – React Frontend:**
```bash
cd front_end && npm run dev
```
→ Mở trình duyệt: **http://localhost:5173**

### Chạy với giao diện dòng lệnh (CLI)

```bash
PYTHONPATH=. python main.py
```

### Tại sao cần `PYTHONPATH=.`?

Dự án sử dụng import kiểu `from src.retrieval import Retrieval` (relative package imports). Biến `PYTHONPATH=.` báo cho Python biết thư mục gốc của project là điểm bắt đầu tìm kiếm module. Không có nó, Python sẽ báo `ModuleNotFoundError`.

**Cách tránh phải gõ mỗi lần** – thêm vào shell profile (`~/.zshrc`):
```bash
# Chỉ áp dụng khi ở trong thư mục dự án này
alias runapi="PYTHONPATH=. python main.py --api"
alias runcli="PYTHONPATH=. python main.py"
```

Hoặc tạo file `.env` và dùng `python-dotenv`, hoặc cài dự án dạng package:
```bash
pip install -e .   # cần tạo setup.py hoặc pyproject.toml
```

---

## 🧪 Chạy Evaluation (RAGAS)

1. Điền dữ liệu vào `eval/golden_dataset.json`
2. Đảm bảo đã có `GEMINI_API_KEY` trong `.env`
3. Chạy câu hỏi qua app để tạo log trong MongoDB
4. Chạy evaluation:

```bash
PYTHONPATH=. python eval/evalutation.py
```
→ Kết quả xuất ra `eval/eval_result.csv`

---

## 🏗️ Kiến trúc hệ thống

```
User Query
    │
    ▼
FastAPI (/chat/stream) ─── SSE Streaming ──► React UI
    │
    ▼
Orchestrator
    ├─ 1. Hybrid Search (Milvus) ─── Dense (SBERT) + Sparse (BM25)
    ├─ 2. MMR Rerank + Lost-in-Middle Reorder
    ├─ 3. Context → Redis (session cache)
    ├─ 4. Generation (Qwen2.5-7B GGUF via llama.cpp)
    └─ 5. Log → MongoDB (query + answer + contexts cho RAGAS)
```
