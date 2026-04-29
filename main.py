"""
main.py – Điểm khởi động chính.

Chế độ 1 (CLI):   python main.py
Chế độ 2 (API):   python main.py --api
  → Khởi động FastAPI tại http://localhost:8000
  → Sau đó chạy riêng: cd front_end && npm run dev  (React tại http://localhost:5173)
"""
import sys

def run_cli():
    from database.connection import DatabaseManager
    from src.orchestrator import Orchestrator

    print("=" * 60)
    print("  HỆ THỐNG TRỢ LÝ PHÁP LUẬT RAG (MongoDB + Redis + Milvus)")
    print("=" * 60)

    print("\n[1] Khởi tạo kết nối Database (Redis & MongoDB)...")
    DatabaseManager().connect()

    print("\n[2] Khởi tạo Orchestrator Pipeline (LLM GGUF & Embedding)...")
    orchestrator = Orchestrator()

    print("\n" + "=" * 60)
    print("Gõ 'exit' để thoát.")
    print("=" * 60)

    session_id = "user_cli_session_1"
    while True:
        try:
            query = input("\nBạn: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "thoát"):
                print("Hẹn gặp lại!")
                break
            print("\nTrợ lý đang suy nghĩ...")
            print(f"\nTrợ lý:\n{orchestrator.ask(query, session_id)}")
        except KeyboardInterrupt:
            print("\nHẹn gặp lại!")
            break
        except Exception as e:
            print(f"\n[Lỗi] {e}")


def run_api():
    import uvicorn
    print("=" * 60)
    print("  FastAPI Backend  →  http://localhost:8000")
    print("  React Frontend   →  cd front_end && npm run dev")
    print("=" * 60)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    if "--api" in sys.argv:
        run_api()
    else:
        run_cli()

