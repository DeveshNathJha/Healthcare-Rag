import fitz  # PyMuPDF
import os
import json
import uuid
import pandas as pd
from img2table.document import PDF
from img2table.ocr import EasyOCR
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.utils import get_logger

logger = get_logger(__name__)

# Cache the EasyOCR tool so it doesn't load into memory on every request
_ocr_tool = None

def get_ocr_tool():
    global _ocr_tool
    if _ocr_tool is None:
        logger.info("Initializing img2table EasyOCR backend...")
        # using the same default as processor.py, if english
        _ocr_tool = EasyOCR(lang=["en"])
    return _ocr_tool

def detect_if_scanned(file_path: str) -> bool:
    """Uses PyMuPDF to check if a PDF has selectable text."""
    try:
        doc = fitz.open(file_path)
        text_length = 0
        for page in doc:
            text_length += len(page.get_text())
            if text_length > 100: # Found enough digital text
                return False
        return True
    except Exception as e:
        logger.error(f"Error detecting if PDF is scanned: {e}")
        return True # Fallback to scanned (OCR) to be safe

def convert_pdf_to_excel(pdf_path: str, excel_path: str) -> bool:
    """
    Extracts tables from a PDF (digital or scanned) and writes directly to Excel.
    """
    is_scanned = detect_if_scanned(pdf_path)
    logger.info(f"[TABLE EXTRACT] Generating Excel for {pdf_path} (is_scanned={is_scanned})")
    
    ocr = get_ocr_tool() if is_scanned else None
    
    pdf_doc = PDF(
        pdf_path, 
        pages=None, 
        detect_rotation=is_scanned, 
        pdf_text_extraction=not is_scanned
    )
    
    # Run extraction to excel
    pdf_doc.to_xlsx(
        dest=excel_path,
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=True,
        min_confidence=50
    )
    return True

def validate_with_llm(pdf_path: str, final_excel_path: str):
    """
    1. Extracts raw data to a temporary excel file.
    2. Reads the excel data with pandas.
    3. Sends to Groq Llama-3 to structure into JSON.
    4. Writes the JSON back to the final excel file.
    """
    logger.info(f"[LLM VALIDATION] Starting LLM validation for {pdf_path}")
    
    # 1. Extract to temp Excel via img2table
    temp_excel = f"/tmp/{uuid.uuid4()}_raw.xlsx"
    convert_pdf_to_excel(pdf_path, temp_excel)
    
    # 2. Read with pandas
    try:
        # img2table puts tables in different sheets
        sheets_dict = pd.read_excel(temp_excel, sheet_name=None)
    except Exception as e:
        logger.error(f"[LLM VALIDATION] Failed to read extracted excel: {e}")
        if os.path.exists(temp_excel):
            os.remove(temp_excel)
        raise ValueError("Could not extract any understandable tables from the document.")
        
    # If no tables found, raise error
    if not sheets_dict or all(df.empty for df in sheets_dict.values()):
        if os.path.exists(temp_excel):
            os.remove(temp_excel)
        raise ValueError("No tabular data found in the document to validate.")

    # Convert sheets to a readable string for LLM
    raw_data_str = ""
    for sheet_name, df in sheets_dict.items():
        if not df.empty:
            raw_data_str += f"--- Sheet: {sheet_name} ---\n"
            raw_data_str += df.to_csv(index=False) + "\n\n"

    # 3. LLM Processing
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    template = """You are an expert Data Engineer in the healthcare domain.
I have extracted raw tabular data from a medical document using OCR. The OCR output may be messy, misaligned, or slightly broken.

Please carefully reconstruct the data into a perfectly structured JSON array of objects.
Each object should represent a valid row. Infer the proper column names based on the content (e.g., Patient Name, Dosage, Date, Doctor, etc.).
Fix minor OCR typos, but be very cautious not to alter real clinical facts or numbers.

OUTPUT FORMAT:
Output ONLY a valid JSON Array. No markdown formatting, no explanations, no backticks (```json). Just the raw list.
If there are multiple distinct tables, combine them intelligently into one consistent structure or include a 'Table_Source' key.

RAW DATA:
{raw_data}
"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    logger.info("[LLM VALIDATION] Querying Groq Llama-3 to structure table...")
    result_str = chain.invoke({"raw_data": raw_data_str})
    
    # Clean up markdown code blocks if LLM adds them
    if result_str.startswith("```json"):
        result_str = result_str[7:]
    if result_str.endswith("```"):
        result_str = result_str[:-3]
    result_str = result_str.strip()
    
    # 4. JSON -> final Excel
    try:
        data_list = json.loads(result_str)
        if isinstance(data_list, dict):
            # Try to grab the first useful looking list if they wrapped it in a dict
            for v in data_list.values():
                if isinstance(v, list):
                    data_list = v
                    break
            else:
                data_list = [data_list] # Last resort
        elif not isinstance(data_list, list):
            data_list = [data_list] 
            
        final_df = pd.DataFrame(data_list)
        final_df.to_excel(final_excel_path, index=False)
        logger.info(f"[LLM VALIDATION] Successfully saved validated excel to {final_excel_path}")
    except json.JSONDecodeError as e:
        logger.error(f"[LLM VALIDATION] LLM output was not valid JSON: {e}\nOutput was: {result_str[:200]}...")
        # Fallback to the raw extraction by copying the temp file over
        import shutil
        shutil.copy(temp_excel, final_excel_path)
    finally:
        if os.path.exists(temp_excel):
            os.remove(temp_excel)
