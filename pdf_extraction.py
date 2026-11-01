# ============================================================
# COMPLETE PAKISTANI LAW REPORT PROCESSING PIPELINE
# (PDF → TXT → STRUCTURED JSON → CLEANED JSON)
# ============================================================

import os
import re
import json
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF

import config as app_config


# ============================================================
# PATH CONFIGURATION
# ============================================================

PDF_INPUT_DIRECTORY = str(app_config.PDF_DIRECTORY)
TEXT_OUTPUT_DIRECTORY = str(app_config.TEXT_DIRECTORY)

STRUCTURED_JSON_OUTPUT = str(app_config.STRUCTURED_JSON)
CLEANED_JSON_OUTPUT = str(app_config.CLEANED_JSON)

os.makedirs(TEXT_OUTPUT_DIRECTORY, exist_ok=True)

OCR_ENABLED = os.getenv("LEGAL_RAG_OCR", "true").lower() in {"1", "true", "yes", "on"}
OCR_WORKERS = int(os.getenv("LEGAL_RAG_OCR_WORKERS", "6"))
OCR_DPI = int(os.getenv("LEGAL_RAG_OCR_DPI", "250"))
OCR_MIN_CHARS = int(os.getenv("LEGAL_RAG_OCR_MIN_CHARS", "20"))
SKIP_EXISTING = os.getenv("LEGAL_RAG_SKIP_EXISTING_TEXT", "true").lower() in {"1", "true", "yes", "on"}

try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


# ============================================================
# STEP 1 — PDF TO TXT EXTRACTION
# ============================================================

def clean_pdf_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+(\n)', r'\1', text)
    return text.strip()


def ocr_page(page) -> str:
    """Render a PDF page and OCR it with the local Tesseract install."""
    fd, image_path = tempfile.mkstemp(suffix=".png", prefix="phc_ocr_")
    os.close(fd)
    try:
        page.get_pixmap(
            dpi=OCR_DPI,
            colorspace=fitz.csGRAY,
            alpha=False,
        ).save(image_path)
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


def extract_pdf(pdf_path: str):
    """Extract one PDF, falling back to OCR for pages without a text layer."""
    filename = os.path.basename(pdf_path)
    page_count = 0
    ocr_pages = 0
    parts = []

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        for page in doc:
            page_text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE,
            )
            if OCR_ENABLED and len(page_text.strip()) < OCR_MIN_CHARS:
                ocr_text = ocr_page(page)
                if len(ocr_text) > len(page_text.strip()):
                    page_text = ocr_text + "\n"
                    ocr_pages += 1
            parts.append(page_text)
        doc.close()
        text = clean_pdf_text("\n".join(parts))
        return filename, text, page_count, ocr_pages, None
    except Exception as exc:
        try:
            doc.close()
        except Exception:
            pass
        return filename, "", page_count, ocr_pages, str(exc)


def extract_pdfs_to_txt():
    print("\n========== STEP 1: PDF → TXT EXTRACTION ==========")
    print(f"Input:  {PDF_INPUT_DIRECTORY}")
    print(f"Output: {TEXT_OUTPUT_DIRECTORY}")
    print(f"OCR:    {'enabled' if OCR_ENABLED else 'disabled'} | workers={OCR_WORKERS} | dpi={OCR_DPI}")

    pdf_files = sorted(
        filename for filename in os.listdir(PDF_INPUT_DIRECTORY)
        if filename.lower().endswith(".pdf")
    )
    pending = []
    skipped = 0
    for filename in pdf_files:
        output_path = os.path.join(
            TEXT_OUTPUT_DIRECTORY,
            os.path.splitext(filename)[0] + ".txt",
        )
        if SKIP_EXISTING and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            skipped += 1
            continue
        pending.append(filename)

    print(f"Found {len(pdf_files)} PDFs | {skipped} already extracted | {len(pending)} to process")
    if not pending:
        return

    started = time.time()
    extracted = 0
    failed = []
    empty = []
    total_pages = 0
    total_chars = 0
    total_ocr_pages = 0

    with ProcessPoolExecutor(max_workers=OCR_WORKERS) as executor:
        futures = {
            executor.submit(extract_pdf, os.path.join(PDF_INPUT_DIRECTORY, filename)): filename
            for filename in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                filename, text, page_count, ocr_pages, error = future.result()
            except Exception as exc:
                filename = futures[future]
                text, page_count, ocr_pages, error = "", 0, 0, str(exc)

            if error:
                failed.append((filename, error))
                print(f"❌ {filename}: {error}")
            elif not text:
                empty.append(filename)
            else:
                output_path = os.path.join(
                    TEXT_OUTPUT_DIRECTORY,
                    os.path.splitext(filename)[0] + ".txt",
                )
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                extracted += 1

            total_pages += page_count
            total_chars += len(text)
            total_ocr_pages += ocr_pages

            if completed % 25 == 0 or completed == len(pending):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0
                remaining = (len(pending) - completed) / rate if rate else 0
                print(
                    f"Progress {completed}/{len(pending)} | "
                    f"extracted={extracted} ocr_pages={total_ocr_pages} "
                    f"| {rate:.2f} files/s | ETA {remaining/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - started
    report = {
        "pdfs": len(pdf_files),
        "skipped": skipped,
        "pending": len(pending),
        "extracted": extracted,
        "failed": len(failed),
        "empty": len(empty),
        "pages": total_pages,
        "chars": total_chars,
        "ocr_pages": total_ocr_pages,
        "elapsed_s": round(elapsed, 1),
        "failed_files": failed,
        "empty_files": empty,
    }
    report_path = os.path.join(TEXT_OUTPUT_DIRECTORY, "_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(
        f"PDF Extraction Completed | extracted={extracted} failed={len(failed)} "
        f"empty={len(empty)} pages={total_pages} chars={total_chars} "
        f"ocr_pages={total_ocr_pages} elapsed={elapsed/60:.1f} min"
    )
    print(f"Report: {report_path}\n")


# ============================================================
# STEP 2 — METADATA EXTRACTION
# ============================================================

def extract_text(file_path):

    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()
        return full_text.strip()

    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    return ""


def extract_metadata(text, filename):

    metadata = {
        "file_name": filename,
        "decision_date": None,
        "judge_name": None,
        "appellant": None,
        "respondent": None,
        "court": None,
        "decision_status": None,
    }

    stem = os.path.splitext(filename)[0]
    parties = re.split(r"\s+(?:vs\.?|versus)\s+", stem, maxsplit=1, flags=re.IGNORECASE)
    if len(parties) == 2:
        metadata["appellant"] = parties[0].strip() or None
        metadata["respondent"] = parties[1].strip() or None

    court_match = re.search(r'\[(.*?)\]', text)
    if court_match:
        metadata["court"] = court_match.group(1).strip()
    if not metadata["court"]:
        court_match = re.search(
            r'IN\s+THE\s+([A-Z][A-Z\s,]+COURT[A-Z\s,]*)',
            text,
        )
        if court_match:
            metadata["court"] = re.sub(r'\s+', ' ', court_match.group(1)).strip(" ,")
    if not metadata["court"] and re.search(r'PESHAWAR\s+HIGH\s+COURT', text, re.IGNORECASE):
        metadata["court"] = "Peshawar High Court"

    judge_match = re.search(r'Before\s+(.*?)\n', text)
    if judge_match:
        metadata["judge_name"] = judge_match.group(1).strip()
    if not metadata["judge_name"]:
        judge_match = re.search(
            r'(?:Mr\.?|Mrs\.?|Ms\.?)\s*Justice\s+([A-Z][A-Za-z.\s]{2,60})',
            text,
        )
        if judge_match:
            metadata["judge_name"] = judge_match.group(1).strip(" .,")

    appellant_match = re.search(r'^\s*(.*?)---Appellant', text, re.MULTILINE)
    if appellant_match and not metadata["appellant"]:
        metadata["appellant"] = appellant_match.group(1).strip()

    respondent_match = re.search(r'^\s*(.*?)---Respondents?', text, re.MULTILINE)
    if respondent_match and not metadata["respondent"]:
        metadata["respondent"] = respondent_match.group(1).strip()

    date_match = re.search(
        r'(?:decided on|date of (?:hearing|judgment|decision)|announced on)\s*[:\-]?\s*'
        r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        text,
        re.IGNORECASE,
    )
    if date_match:
        metadata["decision_date"] = date_match.group(1).strip()

    decision_patterns = [
        "Bail granted",
        "Bail grante",
        "Bail refused",
        "Bail allowed",
        "Bail dismissed",
        "Petition dismissed",
        "Petition accepted",
        "Application dismissed",
        "Revision dismissed",
        "Appeal dismissed",
        "Appeal allowed",
        "Sentence reduced",
        "allowed",
        "dismissed",
        "acquitted",
        "convicted",
        "granted",
        "refused",
    ]

    text_lower = text.lower()
    for status in decision_patterns:
        if re.search(rf'\b{re.escape(status)}\b', text_lower):
            metadata["decision_status"] = status
            break

    return metadata


def create_structured_json():

    print("\n========== STEP 2: METADATA EXTRACTION ==========")

    all_documents = []

    files = os.listdir(TEXT_OUTPUT_DIRECTORY)
    print(f"Found {len(files)} files")

    for index, file in enumerate(files, 1):

        if not file.lower().endswith((".pdf", ".txt")):
            continue

        file_path = os.path.join(TEXT_OUTPUT_DIRECTORY, file)
        print(f"\nProcessing {index}: {file}")

        text = extract_text(file_path)

        if not text:
            print("⚠ No text extracted")
            continue

        metadata = extract_metadata(text, file)

        document_entry = {
            "metadata": metadata,
            "text": text
        }

        all_documents.append(document_entry)

        print("✓ Metadata Extracted")

    if all_documents:
        with open(STRUCTURED_JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False)

        print("\nStructured JSON saved.")
    else:
        print("❌ No documents processed.")

    return all_documents


# ============================================================
# STEP 3 — CLEAN JSON TEXT
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return text

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_cleaned_json():

    print("\n========== STEP 3: CLEANING TEXT ==========")

    if not os.path.exists(STRUCTURED_JSON_OUTPUT):
        print("❌ Structured JSON not found.")
        return

    with open(STRUCTURED_JSON_OUTPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "text" in item:
            item["text"] = clean_text(item["text"])

    with open(CLEANED_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✓ Cleaned JSON saved.")


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PAKISTANI LAW REPORT COMPLETE PROCESSING SYSTEM")
    print("=" * 70)

    extract_pdfs_to_txt()
    create_structured_json()
    create_cleaned_json()

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")