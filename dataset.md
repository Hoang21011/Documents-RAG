# Chi tiết Dataset - NEU Connect RAG

Tài liệu này mô tả cấu trúc, định dạng và quy trình xử lý dữ liệu chi tiết từ các file thô đến khi đưa vào hệ thống RAG, dựa trên các phân tích EDA và kỹ thuật tiền xử lý đã triển khai.

---

## 1. Quy trình Biến đổi Dữ liệu (Data Pipeline)
Dữ liệu trải qua các giai đoạn xử lý nghiêm ngặt được mô tả trong folder `eda-and-preprocessing`:

1.  **Giai đoạn thô (`data/raw/`)**: 
    *   Chứa các file PDF gốc từ NEU và Bộ GD&ĐT.
    *   Sử dụng dataset `th1nhng0/vietnamese-legal-documents` (phiên bản metadata) làm nguồn bổ trợ cho các văn bản pháp quy toàn quốc.

2.  **Xử lý hình ảnh & OCR (`preprocessing_picture.ipynb`)**:
    *   Đối với các file PDF dạng scan hoặc hình ảnh, hệ thống sử dụng **Tesseract OCR** (ngôn ngữ: `vie`).
    *   **Kỹ thuật xử lý ảnh (OpenCV)**:
        *   Chuyển đổi sang ảnh xám (Grayscale).
        *   Áp dụng **Otsu's Thresholding** (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`) để tách nền và chữ tối ưu hơn so với threshold cố định.
    *   Kết quả được lưu vào `data/extracted/` dưới dạng `.txt`.

3.  **Tiền xử lý & Làm sạch (`preprocessing.ipynb`)**:
    *   **Lọc dữ liệu**: Chỉ giữ lại các văn bản thuộc lĩnh vực "Giáo dục và đào tạo", trạng thái "Còn hiệu lực" và phạm vi "Toàn quốc".
    *   **Chuẩn hóa**: Chuyển đổi định dạng ngày tháng (`ngay_ban_hanh`), xử lý giá trị thiếu (loại bỏ cột `thong_tin_ap_dung`).
    *   **Loại bỏ nhiễu**: Loại bỏ các văn bản từ nguồn không hợp lệ hoặc đã bị bãi bỏ.

4.  **Cấu trúc hóa & Chunking**:
    *   Văn bản sau khi làm sạch được tách thành các chunk (**800 - 1000 ký tự**) với độ gối đầu (**Overlap 100-150 ký tự**).
    *   Lưu trữ cuối cùng tại `data/NEU.json` và `data/GDDT.json`.

---

## 2. EDA & Thống kê Dữ liệu (EDA Insights)
Dựa trên phân tích tại `preprocessing.ipynb`:

*   **Phân phối thời gian**: Các văn bản pháp quy tập trung nhiều nhất vào giai đoạn từ năm 2006 đến nay.
*   **Loại văn bản**: Chủ yếu là **Quyết định**, **Thông tư**, và **Nghị định**.
*   **Chất lượng trích xuất**: Việc áp dụng Otsu's Thresholding giúp giảm tỷ lệ lỗi ký tự (noise) trong quá trình OCR đáng kể đối với các văn bản cũ hoặc bản scan mờ.

---

## 3. Định dạng Dữ liệu Tri thức (Knowledge Base)

### Cấu trúc file JSON (`data/NEU.json`)
```json
{
    "chunk_id": "QD1299_chunk_0",
    "content": "Nội dung văn bản quy chế...",
    "metadata": {
        "title": "Tên quy định/quyết định",
        "so_ky_hieu": "1299/QĐ-ĐHKTQD",
        "ngay_ban_hanh": "15/07/2021",
        "loai_van_ban": "QUYẾT ĐỊNH",
        "linh_vuc": "Giáo dục và Đào tạo",
        "source": "QD1299_quy_dinh_giao_trinh.txt",
        "pham_vi": "NEU/Kinh tế quốc dân"
    }
}
```

---

## 4. Bộ dữ liệu Đánh giá (Golden Dataset)
Nằm tại `eval/golden_dataset.json`, chứa các cặp câu hỏi-đáp thực tế để kiểm tra độ chính xác của Pipeline sau khi qua bước trích xuất và chunking.

---

## 5. Danh mục Dữ liệu
*   **NEU_docs**: 32+ văn bản quy chế nội bộ (PDF/TXT).
*   **GDDT_docs**: Văn bản từ Bộ GD&ĐT (PDF/TXT).
*   **Pháp lý**: Metadata từ dataset pháp luật Việt Nam (hơn 4MB CSV).
