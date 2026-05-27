import os
import subprocess
import sys
import webbrowser
import time
from threading import Thread

def run_project():
    env_name = "rag-health"
    
    # 1. Check if virtual environment exists
    if not os.path.exists(env_name):
        print(f"Environment '{env_name}' not found. Please run your setup script first.")
        return

    # Determine path to python and uvicorn based on OS
    if os.name == 'nt':  # Windows
        uvicorn_exe = os.path.join(env_name, "Scripts", "uvicorn.exe")
    else:  # Linux/Mac
        uvicorn_exe = os.path.join(env_name, "bin", "uvicorn")

    print(f"--- Starting Advanced Clinical Healthcare RAG (Backend & Frontend) ---")
    
    processes = []
    
    # 2. Start Backend process
    try:
        backend_proc = subprocess.Popen(
            [uvicorn_exe, "app.main:app", "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(backend_proc)
        print(" Backend Server triggered (FastAPI running on http://127.0.0.1:8000)")
    except Exception as e:
        print(f" Failed to start Backend Server: {e}")
        return

    # 3. Start Frontend process
    try:
        if os.name == 'nt':
            frontend_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd="frontend",
                shell=True
            )
        else:
            # On Linux, run inside NVM v22 environment
            nvm_init = "source ~/.nvm/nvm.sh && nvm use v22 && npm run dev"
            frontend_proc = subprocess.Popen(
                ["bash", "-c", nvm_init],
                cwd="frontend",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        processes.append(frontend_proc)
        print(" Frontend Dev Server triggered (Vite running on http://127.0.0.1:5173)")
    except Exception as e:
        print(f"️ Warning: Could not trigger frontend server automatically: {e}")
        print("Please run manually: cd frontend && npm run dev")

    # 4. Open dashboard in browser after a short delay
    def open_browser():
        time.sleep(4)  # Wait for servers to boot
        print(" Opening Healthcare Assistant in your browser...")
        webbrowser.open("http://127.0.0.1:5173")

    Thread(target=open_browser).start()

    # 5. Monitor and print output/errors
    try:
        def log_stream(proc, label):
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                # Only log critical lines or server boot statements
                if "INFO" in line or "error" in line.lower() or "warning" in line.lower() or "Uvicorn" in line or "Local:" in line:
                    print(f"[{label}] {line.strip()}")

        Thread(target=log_stream, args=(backend_proc, "Backend")).start()
        
        print("\nPress Ctrl+C to terminate both servers safely.\n")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping all Healthcare RAG servers gracefully...")
    finally:
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass
        print(" All processes shutdown successfully.")

if __name__ == "__main__":
    run_project()