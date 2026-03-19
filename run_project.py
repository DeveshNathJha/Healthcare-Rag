import os
import subprocess
import sys
import webbrowser
import time

def run_project():
    env_name = "rag-health"
    
    # 1. Check if virtual environment exists
    if not os.path.exists(env_name):
        print(f"Environment '{env_name}' not found. Please run your setup script first.")
        return

    # 2. Determine path to python and uvicorn based on OS
    if os.name == 'nt':  # Windows
        python_exe = os.path.join(env_name, "Scripts", "python.exe")
        uvicorn_exe = os.path.join(env_name, "Scripts", "uvicorn.exe")
    else:  # Linux/Mac
        python_exe = os.path.join(env_name, "bin", "python")
        uvicorn_exe = os.path.join(env_name, "bin", "uvicorn")

    print(f"--- Starting Healthcare RAG System ---")
    
    # 3. Open Swagger UI in browser after a short delay
    def open_browser():
        time.sleep(3) # Wait for server to boot
        print("Opening Documentation (Swagger UI) in browser...")
        webbrowser.open("http://127.0.0.1:8000/docs")

    # 4. Start the server
    try:
        # Start browser thread equivalent logic
        from threading import Thread
        Thread(target=open_browser).start()

        # Run Uvicorn
        subprocess.run([uvicorn_exe, "app.main:app", "--reload"])
        
    except KeyboardInterrupt:
        print("\nStopping server...")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_project()