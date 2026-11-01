import json
import re
import uuid

import config as app_config

# ==============================
# SETTINGS
# ==============================

INPUT_PATH = str(app_config.CLEANED_JSON)
OUTPUT_PATH = str(app_config.CHUNKED_JSON)

MAX_WORDS = app_config.CHUNK_MAX_WORDS
OVERLAP_PERCENT = app_config.CHUNK_OVERLAP_PERCENT


# ==============================
# HELPER FUNCTIONS
# ==============================

def extract_case_number_from_text(text):
    """Extract case number from the text itself"""
    patterns = [
        r'(?:Accountability Appeal|Appeal|Petition|Case)\s+No\.?\s*(\d+\s+of\s+\d{4})',
        r'([A-Z]{2,}\s*\d+\s+of\s+\d{4})',
        r'(\d+\s+[A-Z]{3,}\s+\d{4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "unknown_case"


def detect_paragraphs(text):
    """
    Detect paragraphs starting with numbering like: 1. , 2. , 3.
    """
    pattern = r'(\d+\.\s+.*?)(?=\s*\d+\.\s+|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches
    else:
        # Fallback: split by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [text]


def pack_paragraphs(paragraphs, max_words):
    """Merge adjacent short paragraphs so chunks are not dominated by headings."""
    packed = []
    buffer = []
    buffer_words = 0

    for paragraph in paragraphs:
        words = len(paragraph.split())
        if buffer and buffer_words + words > max_words:
            packed.append("\n".join(buffer))
            buffer = []
            buffer_words = 0
        buffer.append(paragraph)
        buffer_words += words

    if buffer:
        packed.append("\n".join(buffer))
    return packed


def dynamic_split(paragraph, max_words=None, overlap_percent=None):
    """
    Split paragraph into chunks with overlap
    """
    max_words = max_words or MAX_WORDS
    overlap_percent = OVERLAP_PERCENT if overlap_percent is None else overlap_percent
    words = paragraph.split()
    word_count = len(words)

    if word_count <= max_words:
        return [paragraph]

    chunks = []
    overlap = int(max_words * overlap_percent)
    step = max_words - overlap

    for start in range(0, word_count, step):
        end = start + max_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if end >= word_count:
            break

    return chunks


def detect_chunk_type(text, paragraph_number):
    """
    Detect chunk type based on content
    """
    text_lower = text.lower()

    # Specific type indicators
    type_indicators = {
        "headnote": ["headnote", "ylr", "plr", "scmr", "reported"],
        "facts": ["facts of the case", "brief facts", "factual background"],
        "arguments": ["learned counsel", "contended", "submitted", "argued"],
        "findings": ["held:", "we find", "findings", "observed"],
        "reasoning": ["reasons", "analysis", "discussion"],
        "judgment": ["judgment", "order", "decision"],
        "legal_provision": ["section", "article", "ordinance", "act"],
        "citation": ["reported as", "citation"],
    }

    for chunk_type, indicators in type_indicators.items():
        for indicator in indicators:
            if indicator in text_lower:
                return chunk_type

    # Based on paragraph number
    if paragraph_number.isdigit():
        para_num = int(paragraph_number)
        if para_num <= 2:
            return "headnote"
        elif para_num <= 5:
            return "facts"
        else:
            return "body"

    return "general"


def clean_field(value):
    """Clean and normalize field values"""
    if value is None:
        return "unknown"

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "unknown"

    return str(value)


# ==============================
# MAIN PROCESS
# ==============================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_chunks = []

for doc_idx, doc in enumerate(data):

    # Extract text
    full_text = doc.get("text", "")

    # Extract metadata from the document (already present in JSON)
    existing_metadata = doc.get("metadata", {})

    # Build document-level metadata from existing extraction
    doc_metadata = {
        "file_name": clean_field(existing_metadata.get("file_name")),
        "decision_date": clean_field(existing_metadata.get("decision_date")),
        "judge_name": clean_field(existing_metadata.get("judge_name")),
        "appellant": clean_field(existing_metadata.get("appellant")),
        "respondent": clean_field(existing_metadata.get("respondent")),
        "court": clean_field(existing_metadata.get("court")),
        "decision_status": clean_field(existing_metadata.get("decision_status")),
    }

    # Try to extract case number from text if not in metadata
    case_number = extract_case_number_from_text(full_text)

    # Detect and merge paragraphs before splitting into bounded chunks
    paragraphs = pack_paragraphs(detect_paragraphs(full_text), MAX_WORDS)

    for para_idx, para in enumerate(paragraphs):

        # Extract paragraph number
        para_match = re.match(r'(\d+)\.', para.strip())
        paragraph_number = para_match.group(1) if para_match else str(para_idx + 1)

        # Split into chunks
        split_chunks = dynamic_split(para, MAX_WORDS, OVERLAP_PERCENT)

        for chunk_idx, chunk_text in enumerate(split_chunks):

            chunk_id = str(uuid.uuid4())

            chunk_data = {
                "chunk_id": chunk_id,
                "case_number": case_number,
                "text": chunk_text,
                "metadata": {
                    # Original extracted metadata
                    "file_name": doc_metadata["file_name"],
                    "decision_date": doc_metadata["decision_date"],
                    "judge_name": doc_metadata["judge_name"],
                    "appellant": doc_metadata["appellant"],
                    "respondent": doc_metadata["respondent"],
                    "court": doc_metadata["court"],
                    "decision_status": doc_metadata["decision_status"],

                    # Chunk-specific metadata
                    "chunk_type": detect_chunk_type(chunk_text, paragraph_number),
                    "paragraph_number": paragraph_number,
                    "chunk_index": chunk_idx,
                    "total_chunks_in_paragraph": len(split_chunks),
                    "word_count": len(chunk_text.split()),
                    "document_index": doc_idx,
                }
            }

            all_chunks.append(chunk_data)


# ==============================
# SAVE OUTPUT
# ==============================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4, ensure_ascii=False)

print("✅ Structure-aware dynamic chunking with original metadata completed")
print(f"📊 Total Documents Processed: {len(data)}")
print(f"📊 Total Chunks Created: {len(all_chunks)}")
print(f"📁 Saved to: {OUTPUT_PATH}")

# Display sample chunks
if all_chunks:
    print("\n" + "="*80)
    print("📄 SAMPLE CHUNKS")
    print("="*80)

    # Show first chunk
    print("\n🔹 First Chunk:")
    print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))

    # Show a middle chunk if available
    if len(all_chunks) > 10:
        print("\n🔹 Sample Middle Chunk:")
        print(json.dumps(all_chunks[len(all_chunks)//2], indent=2, ensure_ascii=False))

# Statistics
print("\n" + "="*80)
print("📈 STATISTICS")
print("="*80)

# Count chunks by type
chunk_types = {}
for chunk in all_chunks:
    ct = chunk["metadata"]["chunk_type"]
    chunk_types[ct] = chunk_types.get(ct, 0) + 1

# print("\n📊 Chunks by Type:")
# for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
#     print(f"   {chunk_type}: {count}")

# # Word count statistics
# word_counts = [chunk["metadata"]["word_count"] for chunk in all_chunks]
# avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
# print(f"\n📝 Average Words per Chunk: {avg_words:.2f}")
# print(f"📝 Min Words: {min(word_counts) if word_counts else 0}")
# print(f"📝 Max Words: {max(word_counts) if word_counts else 0}")

# Decision status distribution
decision_statuses = {}
for chunk in all_chunks:
    status = chunk["metadata"]["decision_status"]
    decision_statuses[status] = decision_statuses.get(status, 0) + 1

print("\n⚖️ Decision Status Distribution:")
for status, count in sorted(decision_statuses.items(), key=lambda x: x[1], reverse=True):
    print(f"   {status}: {count}")

print("\n" + "="*80)
print("✅ Process completed successfully!")
print("="*80)