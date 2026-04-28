import os
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Đảm bảo PYTHONPATH đúng
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import DatabaseManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

load_dotenv()

def compute_custom_metrics(question, ground_truth, answer, llm):
    """
    Sử dụng LLM as a judge (Gemini) để chấm điểm các Custom Metrics
    Accuracy, Completeness, Coherence, Citation Accuracy (Thang điểm 1-5).
    """
    prompt = f"""
    Bạn là một giám khảo chuyên nghiệp đánh giá chất lượng câu trả lời của một hệ thống RAG pháp luật.
    Dưới đây là thông tin cần thiết:
    - Câu hỏi: {question}
    - Đáp án chuẩn (Ground Truth): {ground_truth}
    - Câu trả lời của hệ thống (Answer): {answer}
    
    Hãy chấm điểm (từ 1 đến 5) cho các tiêu chí sau và trả về kết quả dưới định dạng JSON với các key: 
    "accuracy", "completeness", "coherence", "citation_accuracy".
    Chỉ trả về chuỗi JSON, không giải thích gì thêm, bắt đầu bằng {{ và kết thúc bằng }}.
    
    Tiêu chí:
    1. Accuracy (Độ chính xác): Câu trả lời có thông tin chính xác so với đáp án chuẩn không? (1: Hoàn toàn sai, 5: Hoàn toàn chính xác)
    2. Completeness (Độ đầy đủ): Câu trả lời có bao gồm tất cả các ý trong đáp án chuẩn không? (1: Rất thiếu sót, 5: Rất đầy đủ)
    3. Coherence (Độ mạch lạc): Câu trả lời có dễ hiểu, ngữ pháp chuẩn và logic không? (1: Lộn xộn khó hiểu, 5: Mạch lạc, tự nhiên)
    4. Citation Accuracy (Trích dẫn): Câu trả lời có sử dụng trích dẫn (ví dụ [1], [2]) hợp lý và đúng chỗ không? (1: Không trích dẫn hoặc trích dẫn sai, 5: Trích dẫn rõ ràng, đúng chỗ)
    """
    
    try:
        response = llm.invoke(prompt)
        text = response.content.replace('```json', '').replace('```', '').strip()
        metrics = json.loads(text)
        return metrics
    except Exception as e:
        print(f"Lỗi khi tính custom metrics: {e}")
        return {"accuracy": 0, "completeness": 0, "coherence": 0, "citation_accuracy": 0}

def main():
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_gemini_api_key_here":
        print("LỖI: Vui lòng cấu hình GEMINI_API_KEY trong file .env trước khi chạy Evaluation.")
        return
        
    # 1. Khởi tạo Models
    print("Khởi tạo Gemini 2.5 Flash làm giám khảo RAGAS...")
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
    
    # Embedding model dùng chung của hệ thống
    emb_model_name = os.getenv("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert")
    print(f"Khởi tạo Embedding {emb_model_name}...")
    judge_embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
    
    # 2. Đọc Golden Dataset
    dataset_path = "eval/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Không tìm thấy {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    # 3. Kết nối MongoDB để lấy Log thực tế
    db_manager = DatabaseManager()
    db_manager.connect()
    mongo_db = db_manager.get_mongo_db()
    logs_collection = mongo_db['chat_logs']
    
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    custom_metrics_results = []
    
    print(f"Đọc dữ liệu từ MongoDB cho {len(golden_data)} câu hỏi...")
    
    for item in golden_data:
        q = item["question"]
        gt = item["expected_answer"]
        
        # Tìm log mới nhất cho câu hỏi này
        log = logs_collection.find_one({"query": q}, sort=[("timestamp", -1)])
        
        if log:
            ans = log.get("answer", "")
            ctxs = log.get("contexts", [])
            if not ctxs:
                print(f"Cảnh báo: Câu hỏi '{q}' không có contexts trong log. Bỏ qua.")
                continue
                
            # Dataset cho RAGAS
            questions.append(q)
            answers.append(ans)
            contexts_list.append(ctxs)
            ground_truths.append(gt) 
            
            # Tính Custom Metrics
            c_metrics = compute_custom_metrics(q, gt, ans, judge_llm)
            custom_metrics_results.append(c_metrics)
        else:
            print(f"Cảnh báo: Chưa có log kết quả cho câu hỏi '{q}' trong MongoDB. Hãy chạy câu hỏi này trên app trước.")
            
    if not questions:
        print("Không có đủ dữ liệu để tính toán RAGAS.")
        return
        
    # 4. Chuẩn bị Dataset cho RAGAS
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # 5. Chạy RAGAS
    print("Bắt đầu chấm điểm RAGAS (có thể mất vài phút)...")
    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
    except Exception as e:
        print(f"Lỗi khi chạy RAGAS: {e}")
        return
        
    # 6. Gộp kết quả và Lưu CSV
    print("Tổng hợp kết quả...")
    df_ragas = ragas_result.to_pandas()
    
    # Gắn Custom Metrics vào DF
    df_custom = pd.DataFrame(custom_metrics_results)
    
    # Merge
    final_df = pd.concat([df_ragas, df_custom], axis=1)
    
    out_path = "eval/eval_result.csv"
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Hoàn thành! Đã xuất báo cáo chi tiết ra: {out_path}")
    print("\n[Trung bình điểm số]")
    # Tính trung bình các cột số
    numeric_cols = final_df.select_dtypes(include='number').columns
    print(final_df[numeric_cols].mean().to_string())

if __name__ == "__main__":
    main()
