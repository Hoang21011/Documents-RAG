import sys
from database.connection import DatabaseManager
from src.orchestrator import Orchestrator

def main():
    print("="*60)
    print("  HỆ THỐNG TRỢ LÝ PHÁP LUẬT RAG (MONGODB + REDIS + MILVUS)")
    print("="*60)
    
    print("\n[1] Đang khởi tạo kết nối Database (Redis & MongoDB)...")
    db_manager = DatabaseManager()
    db_manager.connect()
    
    print("\n[2] Đang khởi tạo Orchestrator Pipeline (LLM GGUF & Embedding)...")
    # Bước này có thể mất vài giây để LLM load lên RAM/VRAM
    orchestrator = Orchestrator()
    
    print("\n" + "="*60)
    print("TẤT CẢ ĐÃ SẴN SÀNG! Gõ 'thoát' hoặc 'exit' để dừng ứng dụng.")
    print("="*60)
    
    session_id = "user_cli_session_1"
    
    while True:
        try:
            query = input("\nBạn: ")
            if not query.strip():
                continue
                
            if query.lower() in ['exit', 'quit', 'thoát']:
                print("Hẹn gặp lại!")
                break
                
            print("\nTrợ lý đang suy nghĩ...")
            # Gọi API RAG hoàn chỉnh
            response = orchestrator.ask(query, session_id)
            print(f"\nTrợ lý:\n{response}")
            
        except KeyboardInterrupt:
            print("\nHẹn gặp lại!")
            break
        except Exception as e:
            print(f"\n[Lỗi] Hệ thống gặp trục trặc: {e}")

if __name__ == "__main__":
    main()
