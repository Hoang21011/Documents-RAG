# Tổng hợp Thuật toán & Kỹ thuật cốt lõi - NEU Connect RAG

Tài liệu này chi tiết các thuật toán được sử dụng để tối ưu hóa khả năng truy xuất, xếp hạng và sinh văn bản trong hệ thống.

---

## 1. Truy xuất Hybrid (Hybrid Retrieval)
Hệ thống kết hợp sức mạnh của tìm kiếm ngữ nghĩa và tìm kiếm từ khóa.

*   **Dense Retrieval (Bi-Encoder)**: 
    *   **Mô hình**: `keepitreal/vietnamese-sbert`.
    *   **Cơ chế**: Chuyển đổi query và văn bản thành vector không gian (embeddings). Sử dụng **Cosine Similarity** trên Milvus để tìm các đoạn có ý nghĩa gần nhất.
*   **Sparse Retrieval (BM25)**:
    *   **Thuật toán**: Best Matching 25 (Okapi BM25).
    *   **Mục đích**: Tìm kiếm chính xác các từ khóa hiếm, thuật ngữ chuyên ngành hoặc mã số Điều/Khoản trong quy chế mà tìm kiếm ngữ nghĩa đôi khi bỏ sót.
*   **Reciprocal Rank Fusion (RRF)**:
    *   **Công thức**: $Score(d) = \sum_{r \in R} \frac{1}{k + r(d)}$
    *   **Mục đích**: Hợp nhất kết quả từ Milvus (Dense) và BM25 (Sparse) mà không cần quan tâm đến thang đo điểm số khác nhau của chúng.

---

## 2. Xếp hạng & Lọc (Reranking)
Sau khi có top-k kết quả, hệ thống tinh lọc lại để đảm bảo thông tin chất lượng nhất.

*   **Maximal Marginal Relevance (MMR)**:
    *   **Mục đích**: Cân bằng giữa **Độ liên quan (Relevance)** và **Độ đa dạng (Diversity)**. 
    *   **Cơ chế**: Tránh việc cung cấp cho LLM 5 đoạn văn bản giống hệt nhau. MMR sẽ chọn đoạn văn bản có điểm số cao nhưng phải khác biệt nhất so với các đoạn đã chọn trước đó.
*   **Lost-in-the-Middle Reordering**:
    *   **Hiện tượng**: LLM thường xử lý tốt thông tin ở đầu và cuối ngữ cảnh, nhưng dễ "quên" thông tin ở giữa (U-shaped performance).
    *   **Giải pháp**: Sắp xếp lại các đoạn văn bản quan trọng nhất (theo điểm MMR) ra hai đầu của context, đẩy các đoạn ít liên quan hơn vào giữa.

---

## 3. Xử lý đa lượt hội thoại (Multi-turn Optimization)
Giải quyết vấn đề trôi ngữ cảnh (Context Drift) khi người dùng hỏi các câu như "Nói rõ hơn", "Cụ thể là gì?".

*   **Query Rewriting (De-contextualization)**:
    *   **Thuật toán**: Sử dụng LLM (Qwen2.5) làm Zero-shot Rewriter.
    *   **Cơ chế**: Phân tích lịch sử chat + Câu hỏi hiện tại -> Sinh ra một **Standalone Query** (Câu hỏi độc lập) chứa đầy đủ chủ ngữ/vị ngữ để tìm kiếm chính xác trong database.

---

## 4. Tối ưu hóa Sinh văn bản (Generation Excellence)
Đảm bảo tốc độ và trải nghiệm người dùng mượt mà trên tài liệu dài.

*   **Flash Attention**: Sử dụng cơ chế tính toán Attention tối ưu hóa bộ nhớ để xử lý Context dài (4096 tokens) nhanh hơn.
*   **KV Caching**: Lưu trữ trạng thái Key-Value của các tokens trước đó để không phải tính toán lại khi sinh token tiếp theo.
*   **Streaming (Server-Sent Events - SSE)**: Trả về token ngay khi vừa sinh xong thay vì đợi cả câu, giúp giảm thời gian phản hồi cảm nhận (Perceived Latency).

---

## 5. Giám sát & Đánh giá (Monitoring)
*   **Latency Tracing**: Đo lường thời gian thực thi của từng module (Retrieval, Rerank, Generation) và log vào hệ thống tập trung.
*   **Guardrails**: Kiểm soát đầu ra để từ chối các câu hỏi ngoài phạm vi quy chế NEU.
