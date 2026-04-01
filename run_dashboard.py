"""
run_dashboard.py — Запускает FastAPI бэкенд одной командой.
Фронтенд запускается отдельно: cd frontend && npm run dev
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Принудительно переключаем вывод на UTF-8 (Windows cp1251 не поддерживает emoji)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Mushroom ML Dashboard - Backend")
    print("  API:   http://localhost:8000")
    print("  WS:    ws://localhost:8000/ws/run")
    print("  Front: http://localhost:5173  (run: cd frontend && npm run dev)")
    print("-" * 55)
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8000"])
