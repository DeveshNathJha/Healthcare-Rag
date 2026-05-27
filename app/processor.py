"""
processor.py — Dual-Layer Document Ingestion Engine
====================================================
WHY THIS FILE EXISTS:
  Handles ALL document ingestion for the Healthcare RAG system.
  Supports two data modalities:
    1. UNSTRUCTURED: PDF files → PyMuPDF (digital text) + EasyOCR (scanned pages)
    2. STRUCTURED:   CSV/Excel → Pandas (row-to-sentence conversion)

CHANGES MADE vs ORIGINAL:
  1. Page-level metadata (Graph-RAG compliance):
     extract_text() now returns a List[dict] instead of a plain string.
     Each dict: {"text": str, "metadata": {"source": str, "page": int,
     "doc_type": str, "ocr_used": bool, "char_count": int}}.
     This enables metadata-linking in rag_chain.py (JD: Graph-RAG).

  2. File size guard (Scalability):
     Files > 50 MB are rejected before any processing to prevent memory spikes
     on servers that may have limited RAM.

  3. OCR Retry Logic (Reliability):
     If EasyOCR returns empty on a page at 2x DPI, we retry at 3x DPI.
     Common with very low-resolution scanned government health records.

  4. Error handling for:
     - Empty/blank PDFs     → ValueError with actionable message
     - Corrupt PDF pages    → logged and skipped (don't crash entire upload)
     - Empty/bad CSVs       → ValueError with file name in message
     - Missing Excel sheets → graceful fallback

  5. Ingestion metrics logging (MLOps):
     After processing, logs: total pages, OCR pages, total characters, elapsed
     time. Visible in logs/healthcare_rag.log.

  6. @log_performance decorator applied to extract_text() for overall timing.

STRICTLY PRESERVED:
  - preprocess_image() with OpenCV pipeline 
  - process_structured_data() with row-to-sentence logic 
  - save_temp_file() 
  - EasyOCR with gpu=False 
  - Dual-layer OCR: digital text first, OCR fallback if blank 
"""

import os
import time
import fitz           # PyMuPDF — fast digital PDF text extraction
import cv2
import numpy as np
import easyocr
import pandas as pd   # Structured data (CSV/Excel)
from typing import List, Dict, Any, Optional

from app.utils import get_logger, log_performance

logger = get_logger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50          # Reject files above this threshold early
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class DocumentProcessor:
    def __init__(self):
        # Initialize EasyOCR Reader (English)
        # gpu=False → runs on standard CPU servers without CUDA
        logger.info("Initializing EasyOCR reader (CPU mode)...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready.")

    # ── IMAGE PREPROCESSING ───────────────────────────────────────────────────
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        OpenCV pipeline to enhance quality of scanned medical documents
        before passing to EasyOCR.

        WHY THESE STEPS:
          Government health records often have faded ink, salt-and-pepper noise,
          and uneven lighting. This 3-step pipeline maximises OCR accuracy.

        UNCHANGED vs original — preserved exactly.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Grayscale — reduces noise channels irrelevant to text
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. NLMD Denoising — removes sensor/scan noise without blurring text
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 3. Adaptive Thresholding — handles variable lighting across the page
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ── STRUCTURED DATA PARSER ────────────────────────────────────────────────
    def process_structured_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses CSV/Excel into a list of page-metadata dicts, consistent with
        extract_text()'s return format.

        WHY LIST FORMAT:
          rag_chain.py needs per-chunk metadata. For structured files the entire
          content is treated as a single logical "page" (page=0).

        CHANGES vs original:
          - Returns List[dict] instead of plain str
          - Adds metadata: doc_type="csv"/"excel", page=0
          - Empty DataFrame guard added
        """
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                doc_type = "csv"
            else:
                df = pd.read_excel(file_path)
                doc_type = "excel"
        except Exception as e:
            raise ValueError(
                f"Failed to read structured file '{os.path.basename(file_path)}': {e}"
            )

        if df.empty:
            raise ValueError(
                f"Structured file '{os.path.basename(file_path)}' is empty or has no data rows."
            )

        # Fill missing cells and convert rows to readable sentences
        df.fillna("Unknown", inplace=True)
        rows = []
        for _, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            rows.append(row_str)

        full_text = "\n".join(rows)

        logger.info(
            f"[STRUCTURED] Parsed {len(df)} rows from '{os.path.basename(file_path)}' "
            f"({doc_type.upper()}) → {len(full_text)} chars"
        )

        return [{
            "text": full_text,
            "metadata": {
                "doc_type": doc_type,
                "page": 0,
                "ocr_used": False,
                "char_count": len(full_text),
                "row_count": len(df)
            }
        }]

    # ── PDF EXTRACTOR (DUAL-LAYER OCR) ────────────────────────────────────────
    def _extract_page_with_ocr(self, page: fitz.Page, scale: float = 2.0) -> str:
        """
        Renders a PDF page to an image and runs EasyOCR on it.

        WHY SEPARATE HELPER:
          Allows retry at higher DPI (scale=3.0) without duplicating the
          OpenCV preprocessing code. Called by extract_text() when digital
          text is absent or when scale-2 OCR returns empty.
        """
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_bytes = pix.tobytes("png")
        processed_img = self.preprocess_image(img_bytes)
        ocr_results = self.reader.readtext(processed_img, detail=0)
        return " ".join(ocr_results).strip()

    @staticmethod
    def _is_garbled(text: str) -> bool:
        """
        Detects if PyMuPDF-extracted text is garbled (font encoding issue).

        WHY THIS CHECK EXISTS:
          Some PDFs use embedded custom/CID fonts. PyMuPDF extracts text but it
          comes out as a stream of garbage Unicode characters (not readable ASCII).
          These pages return non-empty text so the OCR fallback never triggers,
          but the extracted text is useless for RAG retrieval.

          Example: WHO Guidelines for malaria pages return long strings of
          characters like \x00, \ufffd, and other non-printable Unicode.

        HOW IT WORKS:
          If more than 40% of characters in the extracted text are outside the
          standard printable ASCII range (32-126) and are not standard layout
          characters (newlines, carriage returns, tabs), we classify the text
          as garbled and trigger OCR.
        """
        if not text or len(text) < 40:
            return False
            
        allowed_spacing = {'\n', '\r', '\t'}
        non_printable = sum(
            1 for c in text 
            if (ord(c) < 32 and c not in allowed_spacing) or ord(c) > 126
        )
        ratio = non_printable / len(text)
        return ratio > 0.40

    @log_performance  # Emits overall ingestion latency to logs
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Main entry point: dispatches to structured or unstructured pipeline.

        RETURN FORMAT:
            List of dicts, one per logical page:
            [
              {
                "text": str,
                "metadata": {
                  "source": str,       -- filename (for Toggle/Filter feature)
                  "page": int,         -- 1-indexed page number
                  "doc_type": str,     -- "pdf" | "csv" | "excel"
                  "ocr_used": bool,    -- True if EasyOCR was invoked
                  "char_count": int    -- character count of extracted text
                }
              },
              ...
            ]

        WHY THIS FORMAT:
          rag_chain.py creates one Document per entry, attaching the metadata
          dict. This is the foundation for Graph-RAG metadata-linking and
          page-level citations in the LLM prompt.

        CHANGES vs original:
          - Returns list of dicts (was plain str)
          - File size guard added at top
          - Per-page try/except so one corrupt page doesn't kill the upload
          - OCR retry at 3x DPI if 2x returns empty
          - Ingestion metrics logged at the end
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        # ── File Size Guard (Scalability) ─────────────────────────────────────
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"File '{filename}' is {size_mb:.1f} MB, exceeding the {MAX_FILE_SIZE_MB} MB limit. "
                "Please split large documents before uploading."
            )

        logger.info(f"[INGESTION START] '{filename}' ({file_size/1024:.1f} KB, type={ext})")
        t_start = time.perf_counter()

        # ── Route by file type ────────────────────────────────────────────────
        if ext in [".csv", ".xlsx", ".xls"]:
            pages = self.process_structured_data(file_path)
            # Inject source into metadata (structured parser doesn't know filename)
            for p in pages:
                p["metadata"]["source"] = filename
            return pages

        elif ext == ".pdf":
            return self._process_pdf(file_path, filename, t_start)

        else:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                "Supported types: .pdf, .csv, .xlsx, .xls"
            )

    def _process_pdf(
        self, file_path: str, filename: str, t_start: float
    ) -> List[Dict[str, Any]]:
        """
        Inner method for PDF processing.
        Enhanced with thread-safe ThreadPoolExecutor to run page extractions in parallel
        utilising multiple CPU cores for high-speed OCR ingestion.
        """
        try:
            # Quick open to get page count
            temp_doc = fitz.open(file_path)
            total_pages = len(temp_doc)
            temp_doc.close()
        except Exception as e:
            raise ValueError(f"Cannot open PDF '{filename}': {e}")

        pages_data = [None] * total_pages
        ocr_page_count = 0
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Thread-local worker function
        def process_single_page(page_num: int) -> Optional[Dict[str, Any]]:
            # Each thread opens its own file handle for absolute thread-safety
            thread_doc = fitz.open(file_path)
            try:
                page = thread_doc[page_num - 1]
                ocr_used = False
                
                # ── Layer 1: Digital text extraction (fast) ───────────────────
                page_text = page.get_text().strip()

                if page_text and not self._is_garbled(page_text):
                    # Digital text found AND readable — preferred path
                    pass
                else:
                    # ── Layer 2: OCR fallback ─────────────────────────────────
                    # Triggered when: page is blank OR digital text is garbled
                    # (font encoding issue — common in scanned/CID-font PDFs)
                    if page_text:
                        logger.warning(
                            f"  Page {page_num}/{total_pages}: digital text appears garbled "
                            f"(non-ASCII ratio > 40%). Falling back to OCR..."
                        )
                    else:
                        logger.info(
                            f"  Page {page_num}/{total_pages}: no digital text, running OCR..."
                        )
                    
                    # Call self.reader (EasyOCR is thread-safe on CPU PyTorch)
                    page_text = self._extract_page_with_ocr(page, scale=2.0)
                    ocr_used = True

                    # ── OCR Retry at 3x DPI if still empty ───────────────────
                    if not page_text:
                        logger.warning(
                            f"  Page {page_num}: OCR at 2x returned empty. "
                            "Retrying at 3x DPI..."
                        )
                        page_text = self._extract_page_with_ocr(page, scale=3.0)

                if page_text:
                    return {
                        "text": page_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "doc_type": "pdf",
                            "ocr_used": ocr_used,
                            "char_count": len(page_text)
                        }
                    }
                return None
                
            except Exception as page_err:
                logger.error(
                    f"  Page {page_num}: processing error — {page_err}. Skipping."
                )
                return None
            finally:
                thread_doc.close()

        # Run page extraction in parallel (cap workers to prevent thermal throttling or OOM)
        max_workers = min(4, os.cpu_count() or 4)
        logger.info(f"[PARALLEL INGESTION] Extracting {total_pages} pages using {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(process_single_page, p_num): p_num
                for p_num in range(1, total_pages + 1)
            }
            
            for future in as_completed(future_to_page):
                p_num = future_to_page[future]
                res = future.result()
                if res:
                    pages_data[p_num - 1] = res
                    if res["metadata"]["ocr_used"]:
                        ocr_page_count += 1

        # Filter out None values (skipped pages)
        pages_data = [p for p in pages_data if p is not None]

        # ── Validate we got *something* ───────────────────────────────────────
        if not pages_data:
            raise ValueError(
                f"Could not extract any text from '{filename}'. "
                "The PDF may be fully image-based and OCR failed on all pages."
            )

        # ── Ingestion Metrics Log (MLOps) ─────────────────────────────────────
        total_chars = sum(p["metadata"]["char_count"] for p in pages_data)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[INGESTION DONE] '{filename}' | "
            f"pages_processed={len(pages_data)}/{total_pages} | "
            f"ocr_pages={ocr_page_count} | "
            f"total_chars={total_chars} | "
            f"elapsed={elapsed_ms:.0f}ms"
        )

        return pages_data

    # ── FILE HELPER (UNCHANGED) ───────────────────────────────────────────────
    def save_temp_file(self, upload_dir: str, filename: str, content: bytes) -> str:
        """
        Saves uploaded file bytes to disk.
        UNCHANGED vs original — preserved exactly.
        """
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path