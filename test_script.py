import requests
import os
import time

# API Base URL
BASE_URL = "http://127.0.0.1:8000"

# Filenames from your uploaded image
FILES_TO_TEST = [
    "Guideline WHO Guidelines for malaria.pdf",
    "MSample Written History and Physical Examination.doc - UMNwriteup.pdf",
    "Sample-filled-in-MR.pdf"
]

def upload_files():
    """Bari-bari sabhi files ko upload aur index karne ke liye."""
    print("--- Phase 1: Uploading & Indexing Documents ---")
    for filename in FILES_TO_TEST:
        # Check if file exists locally before uploading
        if not os.path.exists(filename):
            print(f"Skipping {filename}: File not found in current directory.")
            continue
            
        print(f"Uploading: {filename}...")
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            print(f"Response: {response.json().get('message')}")
    print("-" * 50)

def list_indexed_files():
    """Dekhne ke liye ki system mein kaunsi files available hain."""
    print("\n--- Phase 2: Listing Indexed Files ---")
    response = requests.get(f"{BASE_URL}/list-files")
    files = response.json().get("files", [])
    print(f"Files in System: {files}")
    return files

def ask_question(question, target_file=None):
    """Sawal puchne ke liye (Toggle support ke saath)."""
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
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # 1. Upload the files first
    upload_files()
    
    # 2. List them to confirm
    indexed_files = list_indexed_files()
    
    if indexed_files:
        # 3. Test Global Search (Toggle OFF)
        # Pooch raha hai malaria ke baare mein (Ye WHO Guidelines se aayega)
        ask_question("What are the recommended treatments for malaria mentioned in the guidelines?")

        # 4. Test Specific Search (Toggle ON)
        # Pooch raha hai patient history (Ye UMNwriteup file se aayega)
        specific_file = "MSample Written History and Physical Examination.doc - UMNwriteup.pdf"
        if specific_file in indexed_files:
            ask_question(
                "What is the patient's chief complaint?", 
                target_file=specific_file
            )
        
        # 5. Test another Specific Search (Toggle ON)
        mr_file = "Sample-filled-in-MR.pdf"
        if mr_file in indexed_files:
            ask_question(
                "Summarize the clinical findings in this medical record.", 
                target_file=mr_file
            )
    else:
        print("No files found to query. Make sure files are in the same folder as this script.")