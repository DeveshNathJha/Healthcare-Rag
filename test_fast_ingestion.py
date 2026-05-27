import requests
import os
import time

BASE_URL = "http://127.0.0.1:8000"

FILES_TO_TEST = [
    "MSample Written History and Physical Examination.doc - UMNwriteup.pdf",
    "Sample-filled-in-MR.pdf"
]

def clear_db():
    print("--- Phase 0: Resetting Vector Store & Cache ---")
    response = requests.post(f"{BASE_URL}/clear-database")
    print("Reset response:", response.json().get("message"))
    print("-" * 50)

def upload_files():
    print("--- Phase 1: Uploading & Indexing Clean Digital Documents ---")
    for filename in FILES_TO_TEST:
        if not os.path.exists(filename):
            print(f"Skipping {filename}: File not found.")
            continue
            
        print(f"Uploading: {filename}...")
        t0 = time.time()
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        t1 = time.time()
        print(f"Response: {response.json().get('message')} (Time: {t1 - t0:.2f}s)")
    print("-" * 50)

def list_indexed_files():
    print("\n--- Phase 2: Listing Indexed Files ---")
    response = requests.get(f"{BASE_URL}/list-files")
    files = response.json().get("files", [])
    print(f"Files in System: {files}")
    return files

def ask_question(question, target_file=None):
    mode = "Specific File" if target_file else "Global (All Files)"
    print(f"\n--- Testing {mode} Search ---")
    print(f"Question: {question}")
    if target_file:
        print(f"Targeting: {target_file}")

    payload = {
        "question": question,
        "target_file": target_file
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/query", json=payload)
    end_time = time.time()

    if response.status_code == 200:
        data = response.json()
        print(f"AI Response (Time: {end_time - start_time:.2f}s):")
        print(f">>> {data['answer']}")
        print("Performance Metadata:")
        print(f" - Model Used: {data.get('model_used')}")
        print(f" - Cache Hit: {data.get('cache_hit')}")
        print(f" - Confidence: {data.get('confidence')}")
        print(f" - Eval Metrics: {data.get('eval_metrics')}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    clear_db()
    upload_files()
    indexed_files = list_indexed_files()
    
    if indexed_files:
        specific_file = "MSample Written History and Physical Examination.doc - UMNwriteup.pdf"
        if specific_file in indexed_files:
            ask_question(
                "What is the patient's chief complaint?", 
                target_file=specific_file
            )
            
            # Ask the same question again to demonstrate fast cache retrieval
            print("\n--- Querying the same question to test Prompt Cache ---")
            ask_question(
                "What is the patient's chief complaint?", 
                target_file=specific_file
            )
        
        mr_file = "Sample-filled-in-MR.pdf"
        if mr_file in indexed_files:
            ask_question(
                "Summarize the clinical findings in this medical record.", 
                target_file=mr_file
            )
    else:
        print("No files found to query.")
