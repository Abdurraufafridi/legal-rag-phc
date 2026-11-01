# AI Lawyer Assistant (Pakistani Legal RAG) — Comprehensive Technical Documentation

**Project:** FYP - AI Lawyer Assistant (Pakistani Legal Judgments RAG)  
**Location:** `/Users/mc/Desktop/FYP code/rag_env`  
**Date:** 2026-10-27  
**Status:** Evaluated and optimized via controlled experiments

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Document Ingestion & Text Extraction](#document-ingestion--text-extraction)
4. [Chunking Strategy](#chunking-strategy)
5. [Embeddings & Vector Representation](#embeddings--vector-representation)
6. [Vector Database (Qdrant)](#vector-database-qdrant)
7. [Retrieval Pipeline](#retrieval-pipeline)
8. [Reranking](#reranking)
9. [Answer Generation](#answer-generation)
10. [Applications (APIs/UI)](#applications-apisui)
11. [Evaluation Framework](#evaluation-framework)
12. [Experiments & Optimization Results](#experiments--optimization-results)
13. [Configuration Reference](#configuration-reference)
14. [Testing](#testing)
15. [Known Limitations](#known-limitations)
16. [Reproducibility](#reproducibility)

---

## 1. Overview

This system implements a Retrieval-Augmented Generation (RAG) pipeline over Pakistani Peshawar High Court criminal judgments. The corpus consists of **74 PDF judgments** (257 pages total) with extracted text, structured metadata, and chunked representations. Retrieval uses hybrid search (dense + sparse) with Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking, document aggregation, and optional filtering. Generation uses OpenRouter-hosted LLMs with grounded prompts constrained to retrieved judgment text.

Key characteristics:
- **Corpus size:** 74 documents, 1,506 word-window chunks + 74 document-level points (1,580 Qdrant points total)
- **Dense embeddings:** `BAAI/bge-m3` (1024-dim, cosine)
- **Sparse embeddings:** TF-IDF (15,000 features, unigrams+bigrams, min_df=2)
- **Vector store:** Local Qdrant at `qdrant_db/`, collection `pak_legal_rag_multi`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
- **Primary app:** `main.py` (FastAPI, richer features), `legal_rag_system.py` (simplified variant)

---

## 2. System Architecture

```
PDFs (judgments for FYP1/) 
  → pdf_extraction.py (PyMuPDF) → FYP1_extracted_text/*.txt
  → metadata extraction → structured_judgments1.json / structured_judgments1_cleaned.json
  → chunk.py (word-window, numbered-paragraph aware) → structure_dynamic_chunked.json
  → embadding.py (BGE-M3 dense + TF-IDF sparse) → Qdrant collection pak_legal_rag_multi
  → Retrieval (rag_core / main.py / legal_rag_system.py): hybrid prefetch + RRF + rerank + aggregate
  → Generation (OpenRouter): summarize/chat with grounded context (truncated to ~3000 chars)
```

Supporting artifacts: `sparse_vectorizer_pak.pkl` (pickled TF-IDF vectorizer), cached models in `~/.cache/huggingface/`.

---

## 3. Document Ingestion & Text Extraction

**Files:** `pdf_extraction.py`, data directories

### Process
1. **Extraction:** `extract_pdfs_to_txt()` reads all PDFs from `PDF_INPUT_DIRECTORY` (`rag_env/judgments for FYP1/`) using PyMuPDF (`fitz`). For each page, text is extracted with `flags=fitz.TEXT_PRESERVE_WHITESPACE`.
2. **Cleaning:** `clean_pdf_text()` normalizes excessive newlines (collapses 3+ to 2) and strips trailing spaces. No OCR is implemented.
3. **Structuring:** `create_structured_json()` reads extracted `.txt` files and applies naive regex-based metadata extraction (court from first bracket pair, judge from `Before` lines, appellant/respondent from `---Appellant/---Respondent`, date from `decided on`, status keywords). Writes `structured_judgments1.json`.
4. **Cleaned structuring:** `create_cleaned_json()` collapses **all** newlines to spaces and normalizes whitespace. Writes `structured_judgments1_cleaned.json`.

### Observations
- 74/74 PDFs represented; total extracted text ~594KB. Corpus is digitally generated (no OCR needed).
- Newline removal in cleaned JSON affects paragraph detection downstream.
- Metadata extraction is heuristic and incomplete for some fields (e.g. many missing dates/appellants).

---

## 4. Chunking Strategy

**File:** `chunk.py` | **Output:** `structure_dynamic_chunked.json` (1,506 chunks)

### Parameters
- `MAX_WORDS = 100`
- `OVERLAP_PERCENT = 0.25` → stride = 75 words
- Unit of measurement: **words** (not tokens)

### Algorithm
1. **Paragraph detection:** `detect_paragraphs()` first attempts regex for numbered paragraphs: `r'(\d+\.\s+.*?)(?=\s*\d+\.\s+|$)'`. If no matches, falls back to splitting on `\n\n`. Because newlines are collapsed in cleaned data, numbered paragraphs are usually the effective unit.
2. **Sliding window:** `dynamic_split()` slides a 100-word window with 75-word stride over paragraph text. Final window may be < 100 words.
3. **Classification:** Heuristic `classify_chunk()` assigns types (headnote, facts, arguments, legal_provision, reasoning, findings, judgment, body).
4. **Metadata enrichment:** Per-document `case_number` extracted via regexes (can match citations), copied to all chunks. Carries file_name, dates/judges/parties/court/status, paragraph/chunk indices, word count.

### Statistics (measured)
- Mean/median words: 71.6/88; 686 chunks exactly 100 words; 84 chunks ≤ 10 words (5.6%)
- 467/1506 chunks have `case_number="unknown_case"` (31%)
- 26 duplicate chunk texts; chunk IDs are random UUID4s (non-deterministic across re-runs)

---

## 5. Embeddings & Vector Representation

**File:** `embadding.py`

### Dense
- Model: `BAAI/bge-m3` (sentence-transformers)
- Dimension: 1024, cosine similarity, `normalize_embeddings=True`
- Indexing `MAX_SEQ_LENGTH = 512` (query-time uses model default 8192)
- Batch size: 4

### Sparse
- `TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=2, sublinear_tf=True)` fit on all chunk texts
- Pickled as `sparse_vectorizer_pak.pkl` (1.13MB)
- Stored as TF-IDF weights in Qdrant (no BM25, no sparse index config)

### Document-level points
For each of the 74 documents, all chunk texts concatenated are embedded as a separate "document-level" point (truncated by 512-token limit). These share `file_name` with chunks and duplicate search space.

### Indexing behavior
- Creates collection if missing; no schema migration/versioning
- Fresh UUID4 per point on each run (risk of duplicates on reindex)
- No dimension checks beyond using computed `embed_dim`

---

## 6. Vector Database (Qdrant)

**Location:** `qdrant_db/` (local)  
**Collection:** `pak_legal_rag_multi`

### Schema (measured)
- **Dense vector:** `dense`, size 1024, distance Cosine
- **Sparse vector:** `sparse` (TF-IDF weights)
- **Total points:** 1,580 (1,506 chunk-level + 74 document-level)
- **Payloads:** include `file_name`, `text`, `chunk_id`, `level`, document metadata, chunk metadata
- **Payload indexes:** none observed
- **Point IDs:** UUID4 (random)

---

## 7. Retrieval Pipeline

**Implementations:** `rag_core.py` (evaluation/reusable), `main.py`, `legal_rag_system.py`

### Hybrid search (`hybrid_search`)
1. Encode query with BGE-M3 (normalized)
2. Transform query with pickled TF-IDF vectorizer
3. Qdrant `query_points` with two prefetches (dense limit 50, sparse limit 50)
4. Fuse with **Reciprocal Rank Fusion (RRF)** → return up to 50 points

### Document aggregation
`aggregate_points()` / `aggregate_by_document()` groups by `file_name`. Document-level points merge with chunks. Can deduplicate by `chunk_id` (optional). Empty/whitespace-only texts ignored.

### Document filtering
`main.py` supports filename-based year range filtering (parses years from patterns like `between/from/-/to` in filename). Documents without years are kept (fallback behavior).

### Ranking modes
- **With reranking (default):** For each candidate document, form `[query, chunk_text]` pairs for all chunks of that document; run cross-encoder `predict()`; take **max** sigmoid probability across chunks as document score; sort descending.
- **Without reranking:** Use retrieval score (best/aggregated) for ranking.

### Candidate controls
- `prefetch_limit` / `candidate_limit`: control hybrid pool size
- `max_chunks_per_doc`: limit chunks passed to reranker per document (ablation)
- `rerank_max_docs`: limit number of docs sent to reranker (ablation)
- `include_document_level`: include/exclude document-level points from aggregation
- `dedupe_chunks`: deduplicate by chunk_id

**Default (main.py behavior):** hybrid prefetch 50+50, RRF, rerank all candidate chunks, include document-level, top-3, threshold 0.75.

---

## 8. Reranking

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformers CrossEncoder)
- **Input:** pairs `[query, chunk_text]` per chunk
- **Scoring:** logits → sigmoid probabilities
- **Aggregation:** max probability over document's chunks
- **Max length:** 512 tokens (configurable)
- **Cost:** dominant latency factor when reranking many chunks; scales with total chunk-candidates

---

## 9. Answer Generation

**Endpoints:** `/api/summarize`, `/api/chat` (OpenRouter)

- **Model:** `mistralai/mixtral-8x7b-instruct` (default)
- **Temperature:** 0.2
- **Max tokens:** 250 (summary), 200 (chat)
- **Context:** Retrieved judgment text truncated to **first 3,000 characters** (`text[:3000]`)
- **Prompting:** Grounded to provided judgment only; refusal for out-of-scope questions
- **Notes:** Naive character truncation (no token budgeting), no chunk-selection optimization before generation

---

## 10. Applications (APIs/UI)

### `main.py` (primary)
- FastAPI app with embedded HTML/JS chat UI
- Auth: HTTP Bearer sessions (in-memory store in codebase style; file persistence under Windows paths in original code - not portable as-is)
- Features: search, summarize, chat, history (last 50), saved judgments
- Retrieval: returns up to 3 documents with score ≥ 0.75
- Uploads: `/api/upload` accepts PDF/DOCX/TXT, embeds first 1000 chars, stores full text in payload, upserts to production collection; retrieval cannot load uploaded files via `load_full_document()` (looks in `FYP1_extracted_text/` only)

### `legal_rag_system.py` (alternative)
- Similar pipeline, simpler behavior
- Returns single best document, threshold 0.60
- In-memory sessions (24h expiry), SHA-256 unsalted passwords

**Security notes:** Hard-coded API keys present in codebase (treat as compromised), `allow_origins=["*"]`, unsalted SHA-256 passwords - not production-safe.

---

## 11. Evaluation Framework

**Location:** `evaluation/`

### Components
- **`dataset.py`**: Loads queries from `evaluation/queries/evaluation_queries.jsonl`, qrels from `evaluation/queries/relevance_judgments.jsonl`. Validates grades (0/1/2), ensures all qrels reference known documents, separates in-domain/out-of-domain.
- **`metrics.py`**: Computes Recall@k, Precision@k, MRR@k, MAP@k, nDCG@k (binary relevance by default). Handles empty rankings and out-of-domain separation.
- **`run_evaluation.py`**: Orchestrates experiments - loads config, initializes `LegalRetriever` from `rag_core`, runs retrieval for all queries with timing, computes metrics at ranking level (cutoff 10) and production level (app top-k), writes JSON results + appends to `experiment_comparison.csv`. Captures environment metadata (packages, platform, collection info, dataset hashes).
- **`rag_core.py`**: Reusable retrieval implementation mirroring app behavior (configurable, testable).

### Evaluation data
- **Queries:** 44 (evaluation_queries.jsonl)
- **Relevance judgments:** 107 (relevance_judgments.jsonl) - binary/graded labels for in-domain queries
- **Out-of-domain:** 4 queries (marked, expected no relevant docs)

### Metrics reported
- **Top-10 ranking:** Recall@5, Recall@10, Precision@5, MRR@10, MAP@10, nDCG@5, nDCG@10
- **Production mode:** Recall@k, Precision@k, nDCG@k at `final_top_k` (default 3) with `threshold` applied
- **Latency:** per-query total/hybrid/rerank/aggregation times (median, p95, mean)

---

## 12. Experiments & Optimization Results

All experiments run with 44 in-domain queries against the existing collection (no reindexing). Results saved in `evaluation/results/`, summarized in `evaluation/results/experiment_comparison.csv`.

### Baseline
**Config:** `baseline_main` - hybrid RRF, prefetch 50+50, rerank all candidate chunks, include document-level points, top-3, threshold 0.75 (replicates `main.py`)

| Metric | Value |
|--------|-------|
| Recall@10 | 0.9450 |
| nDCG@10 | 0.9339 |
| MRR@10 | 0.9875 |
| Production Recall@3 | 0.6879 |
| Production nDCG@3 | 0.8673 |
| Median latency | 6.90s |

### Controlled experiments

| Experiment | Config | Description | Recall@10 | nDCG@10 | Prod Recall@3 | Prod nDCG@3 | Median (s) |
|------------|--------|-------------|-----------|---------|---------------|---------------|------------|
| baseline_main | baseline_main | Original behavior (with doc-level) | 0.9450 | 0.9339 | 0.6879 | 0.8673 | 6.904 |
| exp01 | exp01_chunk_level_only | Exclude document-level points | 0.9408 | 0.9149 | 0.6817 | 0.8515 | 2.321 |
| exp02 | exp02_chunk_level_rerank256 | Chunk-level only, reranker max_length 256 | 0.9408 | 0.9149 | 0.6817 | 0.8515 | 2.321 |
| exp03 | exp03_retrieve_doc_rerank_chunks | Retrieve with doc-level, rerank chunks only | 0.9367 | 0.9116 | 0.6879 | 0.8555 | 2.102 |
| exp04 | exp04_chunk_level_cand30 | Chunk-level, prefetch/candidate 30 | 0.9283 | 0.9122 | 0.6879 | 0.8555 | 1.994 |
| exp05 | exp05_chunk_level_cand20 | Chunk-level, prefetch/candidate 20 | 0.8879 | 0.9002 | 0.6942 | 0.8579 | 1.758 |
| exp06 | exp06_chunk_level_minilm_l2 | Chunk-level, reranker `all-MiniLM-L6-v2` | 0.9346 | 0.8869 | 0.6417 | 0.7468 | 1.629 |
| exp07 | exp07_chunk_level_tinybert_l2 | Chunk-level, reranker `ms-marco-TinyBERT-L-2-v2` | 0.9088 | 0.8994 | 0.6613 | 0.7995 | 0.942 |
| exp08 | exp08_dense_only_rerank | Dense-only retrieval, chunk-level, rerank | 0.9458 | 0.9197 | 0.6817 | 0.8515 | 2.804 |
| exp09 | exp09_sparse_only_rerank | Sparse-only retrieval, chunk-level, rerank | 0.9229 | 0.9070 | 0.6921 | 0.8556 | 3.316 |
| exp10 | exp10_hybrid_dbsf_rerank | Hybrid with DBSF fusion, chunk-level, rerank | 0.9367 | 0.9113 | 0.6817 | 0.8515 | 1.980 |
| exp11 | exp11_hybrid_rrf_no_rerank | Hybrid RRF, chunk-level, no reranking | 0.9229 | 0.9042 | 0.3113 | 0.4128 | 0.447 |

### Key findings
- **Reranking is critical** for production quality: removing it (exp11) drops production nDCG@3 from ~0.852 to ~0.413 despite reasonable top-10 recall.
- **Document-level points hurt efficiency** with minimal/no gain on this corpus; removing them (exp01) cuts latency by ~3x with small recall drop.
- **Smaller candidate pools** (exp04/exp05) reduce latency further; exp05 gives best production recall@3 (0.6942) at cost of lower Recall@10.
- **Dense-only** (exp08) achieves highest Recall@10 (0.9458); sparse contributes complementary signal in hybrid.
- **Faster rerankers** (exp07) are much faster but reduce production quality.
- **DBSF fusion** (exp10) performs similarly to RRF on this dataset.

### Recommended configuration (balanced)
Based on measurements: **exp01/chunk-level hybrid RRF with reranking** offers strong quality with ~3x speedup over baseline. If production recall@3 is prioritized and some top-10 recall acceptable, **exp05** (cand20) is fastest among reasonable options. The original `main.py` behavior (baseline_main) is preserved as the app baseline.

---

## 13. Configuration Reference

Key numerical parameters (see `docs/NUMERICAL_CONFIGURATION.md` for full inventory):

| Category | Parameter | Value | Location |
|----------|-----------|-------|----------|
| Chunking | Window/overlap/stride | 100 words / 25% / 75 words | `chunk.py` |
| Embedding | Model/dim/normalize | `BAAI/bge-m3` / 1024 / true | `embadding.py` |
| Sparse | TF-IDF features/ngrams/min_df | 15000 / (1,2) / 2 | `embadding.py` |
| Retrieval | Prefetch/fusion | 50+50 / RRF (Qdrant) | `main.py`, `legal_rag_system.py`, `rag_core.py` |
| Reranking | Model/max length/agg | `ms-marco-MiniLM-L-6-v2` / 512 / max over chunks | `main.py`, `legal_rag_system.py` |
| Production | main.py top-k/threshold | 3 / 0.75 | `main.py:45-46` |
| Production | legal_rag_system.py top-k/threshold | 1 / 0.60 | `legal_rag_system.py:44-45` |
| Generation | LLM/temp/max tokens/context | mixtral-8x7b-instruct / 0.2 / 250/200 / 3000 chars | both apps |

Evaluation configs in `evaluation/configs/` control all retrieval parameters reproducibly.

---

## 14. Testing

**Location:** `tests/` | **Runner:** unittest (pytest not installed in venv)

| Test file | Tests | Status |
|-----------|-------|--------|
| `test_dataset.py` | 5 | OK |
| `test_metrics.py` | 7 | OK |
| `test_rag_core.py` | 11 | OK |
| **Total** | **23** | **All passing** |

Tests cover dataset validation, metric computations (Recall/MRR/MAP/nDCG/precision), retrieval config roundtrip/validation, aggregation logic, year filtering, and ranking behavior (rerank with max-chunk aggregation, chunk caps, no-rerank path).

Run command:  
```bash
/Users/mc/Desktop/FYP\ code/rag_env/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

---

## 15. Known Limitations

- **Corpus size:** Only 74 judgments - limited generalizability, provisional relevance labels need FYP review
- **Chunking:** Word-based windows, newline-stripped cleaning breaks paragraph structure; 31% chunks have unknown case numbers; many very short chunks
- **Document-level points:** Duplicate search space, truncated representations, add overhead
- **Retrieval:** No metadata filtering beyond filename-year heuristic in main.py; uploads not retrievable by filename loader; no chunk deduplication by default
- **Reranking:** No per-document chunk cap by default (expensive); applied to all chunks of candidates
- **Generation:** Naive 3000-char truncation, no token budgeting, no citation extraction beyond filename
- **Security:** Hard-coded OpenRouter keys, `allow_origins=["*"]`, unsalted SHA-256 passwords, in-memory/file auth issues
- **Reproducibility:** Random UUID4 point IDs, no content hashes/versioning, paths hard-coded to Windows in main.py persistence
- **Evaluation:** Provisional qrels (107 judgments for 44 queries), created as part of this work

---

## 16. Reproducibility

### Environment
- Python: 3.10.14 (venv at `bin/python`)
- Key deps: torch 2.2.2, sentence-transformers 2.7.0, transformers 4.40.2, qdrant-client 1.17.1, scikit-learn 1.4.2, fastapi/uvicorn, openai
- Models cached in `~/.cache/huggingface/` (BGE-M3, cross-encoders)

### Running evaluation
```bash
cd "/Users/mc/Desktop/FYP code/rag_env"
/Users/mc/Desktop/FYP\ code/rag_env/bin/python -m evaluation.run_evaluation \
  --config evaluation/configs/baseline_main.json --name baseline_main
```

### Running tests
```bash
/Users/mc/Desktop/FYP\ code/rag_env/bin/python -m unittest discover -s tests -p 'test*.py'
```

### Key files
- **Results:** `evaluation/results/*.json`, `evaluation/results/experiment_comparison.csv`
- **Configs:** `evaluation/configs/*.json`
- **Audit:** `docs/AUDIT_REPORT.md`, `docs/NUMERICAL_CONFIGURATION.md`
- **This doc:** `docs/TECHNICAL_DOCUMENTATION.md`
