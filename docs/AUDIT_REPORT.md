# Phase 1 Audit Report — Pakistani Legal RAG System

**Audit date:** 2026-10-27
**Project root:** `/Users/mc/Desktop/FYP code/rag_env`
**Auditor:** Engineering audit performed against the files present in the workspace.
**Status:** Read-only audit. No source PDFs, JSON datasets, vectorizer, or Qdrant collections were modified.

---

## 1. Executive summary

This is a compact, single-machine legal retrieval application built around:

- 74 Peshawar High Court criminal judgments (PDF + extracted `.txt` + structured JSON).
- 1,506 paragraph-style chunks plus 74 document-level points, stored in a **local Qdrant** collection.
- `BAAI/bge-m3` dense embeddings (1024 dimensions, cosine) combined with a 15,000-feature TF-IDF sparse representation.
- Qdrant prefetch + **Reciprocal Rank Fusion (RRF)** for hybrid retrieval.
- A `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker over the chunks of candidate documents.
- A FastAPI application with an embedded HTML/JS chat UI, authentication, history, saved judgments, and OpenRouter LLM calls for grounded summarization and Q&A.

The system is functional in the current macOS environment through `legal_rag_system.py`. The newer `main.py` contains the richer multi-result/date-filter behavior but is **not runnable on this machine as written** because every data path and every persistence path is hard-coded to `C:/Users/LENOVO/...`.

There is **no README, no test suite, no evaluation framework, no relevance judgments, and no committed experiment results**. There is also no Git repository, so source revisions cannot be recovered from Git.

The most consequential engineering issues found were:

1. **`main.py` cannot open the existing data/index on this machine** (Windows absolute paths).
2. **A live OpenRouter API key is hard-coded** in both application files, and there is no `.env` file.
3. **No measured retrieval quality**: the system has never had a fixed query set or relevance labels, so changes cannot be validated.
4. **Chunking is based on words, not tokens**, and a prior cleaning step removes all newlines; paragraph detection therefore depends almost entirely on numbered-paragraph patterns. 467/1,506 chunks (31.0%) have `case_number = "unknown_case"`, and citation strings are sometimes mistaken for case numbers.
5. **Document-level points duplicate the search space** and are built from the first ~512 BGE-M3 tokens of concatenated chunks, not from a meaningful full-document representation.
6. **The reranker is applied to every chunk returned by the 50+50 hybrid prefetch**, without a document-aware candidate cap; this is the dominant query cost and can let long judgments dominate.
7. **Uploads are indexed into the production collection but cannot be returned**: the retriever loads text only from `FYP1_extracted_text/<file_name>`, so uploaded files are effectively write-only. Uploads also embed only the first 1,000 characters.
8. **Generation truncates the judgment to the first 3,000 characters** (`text[:3000]`) rather than selecting sections relevant to the question.
9. **`requirements.txt` is stale for Qdrant**: it pins `qdrant-client==1.9.1`, but the working environment has `qdrant-client==1.17.1`. The locally created database was successfully opened with the installed version.
10. **No metadata payload indexes** are created in Qdrant, and there is no index-version/collection-version strategy.

These findings are evidence-based; file/line references are given below.

---

## 2. Environment (measured, not assumed)

| Item | Value |
|---|---|
| OS | macOS (Darwin), Intel-class workstation |
| Project Python | `/Users/mc/Desktop/FYP code/rag_env/bin/python`, Python 3.10.14 |
| Device | CPU only (`torch.cuda.is_available() == False`) |
| torch | 2.2.2 |
| sentence-transformers | 2.7.0 |
| transformers | 4.40.2 |
| qdrant-client | **1.17.1 installed** (requirements.txt pins 1.9.1) |
| scikit-learn | 1.4.2 |
| numpy | 1.26.4 |
| fastapi / uvicorn | 0.111.0 / 0.29.0 |
| openai | 1.30.0 |
| PyMuPDF / PyPDF2 / python-docx | 1.24.3 / 3.0.1 / 1.1.0 |
| Model cache | `~/.cache/huggingface` (BGE-M3 and cross-encoder already cached) |
| Git | **not a Git repository** |

Baseline model smoke test (measured during audit):

- BGE-M3 load: ~7.4 s, dimension 1024, default `max_seq_length` = 8192 at query time.
- BGE-M3 single short query encode: ~1.5 s including first-call overhead.
- Cross-encoder load: ~1.2 s; a two-pair prediction ~0.01 s after load.

---

## 3. Data inventory (measured)

### 3.1 Source documents

| Measure | Value |
|---|---:|
| PDFs in `judgments for FYP1/` | 74 |
| Unique SHA-256 PDFs | 74 (no duplicate PDFs detected) |
| Extracted `.txt` files in `FYP1_extracted_text/` | 74 |
| PDF stems represented in extracted text | 74/74 |
| Empty/short extracted texts (<1,000 bytes) | 0 |
| Total PDF pages | 257 |
| Median / min / max pages per PDF | 3 / 2 / 16 |
| Total text characters extracted directly by PyMuPDF | 594,048 |

The dataset is therefore **74 documents**, not 1,600. The “1,600+” figure in the project brief is not the size of the corpus actually present in this workspace.

### 3.2 Structured data

| File | Records | Size | Notes |
|---|---:|---:|---|
| `structured_judgments1.json` | 74 | 617,480 B | Cleaned? No — raw metadata/text output |
| `structured_judgments1_cleaned.json` | 74 | 607,134 B | All newlines collapsed to spaces |
| `structure_dynamic_chunked.json` | 1,506 | 1,767,875 B | Current chunk file |
| `sparse_vectorizer_pak.pkl` | 1 vectorizer | 1,129,732 B | TF-IDF, 15,000 features |

### 3.3 Qdrant index (measured by opening the collection read-only)

| Property | Value |
|---|---|
| Collection | `pak_legal_rag_multi` |
| Dense vector | `dense`, size 1024, Cosine |
| Sparse vector | `sparse` (no modifier/index config) |
| Total points | **1,580** |
| `level = chunk` points | 1,506 |
| `level = document` points | 74 |
| Distinct `file_name` values | 74 |
| Payload index definitions | none observed in collection meta |

The point count is internally consistent with 1,506 chunks + 74 document-level points. It does **not** indicate 1,580 distinct judgments.

### 3.4 Chunk statistics

| Measure | Value |
|---|---:|
| Chunks | 1,506 |
| Unique chunk IDs | 1,506 |
| Unique chunk texts | 1,480 (26 duplicate texts) |
| Empty chunks | 0 |
| Mean words/chunk | 71.6 |
| Median words/chunk | 88 |
| Min / max words/chunk | 1 / 100 |
| Chunks ≤ 10 words | 84 (5.6%) |
| Chunks ≤ 20 words | 128 (8.5%) |
| Chunks ≤ 50 words | 448 (29.7%) |
| Chunks exactly 100 words | 686 (45.5%) |
| Word-count percentiles (1/5/25/50/75/90/95/99/100) | 1 / 6 / 44 / 88 / 100 / 100 / 100 / 100 / 100 |
| Chunks with `unknown_case` | 467 (31.0%) |

Chunk-type distribution produced by the heuristic classifier:

| Type | Chunks |
|---|---:|
| body | 319 |
| legal_provision | 313 |
| arguments | 266 |
| headnote | 221 |
| judgment | 202 |
| facts | 111 |
| reasoning | 57 |
| findings | 17 |

Missing/unknown chunk metadata:

| Field | Chunks missing/unknown |
|---|---:|
| `file_name` | 0 |
| `decision_date` | 117 |
| `judge_name` | 0 |
| `appellant` | 939 |
| `respondent` | 102 |
| `court` | 126 |
| `decision_status` | 189 |

### 3.5 Dataset fingerprints (SHA-256)

Measured during the audit (first 64 hex characters shown in full):

| Artifact | SHA-256 |
|---|---|
| Source PDF set (ordered SHA-256 hashes, hashed again) | `e35887d4ad0d63f88e8631df710de604ede824daa103443576e3d85757a4a3b3` |
| `structured_judgments1.json` | `a1b80955bc74f02e9a42d5fe58e3341d5234d27b12716191e4c1913e6e1d1b64` |
| `structured_judgments1_cleaned.json` | `f12c6dfef10499332368780221a1f62fa1d65040bfd7e3063f5b0bf405c56c85` |
| `structure_dynamic_chunked.json` | `64f03538edcbe71cc09b78d46ee913c6e98770b2b1346291b01efa9f66f4e8fc` |
| `sparse_vectorizer_pak.pkl` | `69277153e339eaf9964e93dc031b12e17ce21b1d9520f64fcb35e61bf585ca75` |
| Qdrant `meta.json` (collection config) | recorded in evaluation result metadata at run time |

---

## 4. Actual architecture, traced from code

### 4.1 Ingestion and preprocessing (`pdf_extraction.py`)

1. `extract_pdfs_to_txt()` iterates `PDF_INPUT_DIRECTORY` (`rag_env/judgments for FYP1`) and extracts every page with PyMuPDF (`page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)`).
2. `clean_pdf_text()` collapses 3+ newlines to two and strips trailing spaces; it does **not** detect scanned pages, run OCR, deduplicate, or log failures to a file. Exceptions are printed to stdout only.
3. `create_structured_json()` reads the extracted `.txt` files, extracts metadata with naive regexes, and writes `structured_judgments1.json`.
4. `create_cleaned_json()` collapses **all** newlines to spaces and all whitespace runs to one space. This destroys paragraph and page boundaries.
5. Paths are relative (`rag_env/...`) and assume the process is launched from the parent directory. Running from the project root (as this audit did) would look for `rag_env/rag_env/...`.

There is **no OCR** path. The corpus is digitally generated PDFs in this workspace, so OCR was not required for these 74 documents, but the code has no detection or fallback for scanned documents.

Metadata extraction weaknesses:

- `court` = first bracket pair in the text; on judgments with a citation first, this can be wrong.
- `judge_name` requires a line beginning `Before`, which depends on preserved newlines.
- `appellant`/`respondent` depend on `---Appellant` / `---Respondent` line patterns.
- `decision_date` requires `decided on ...`; many nice values are missing.
- `decision_status` is a first-match keyword list; the code contains the typo `"Bail grante"` (observed as a literal value in output) and can misclassify.
- Case number is not stored at document level at all; it is re-extracted later from the entire document text.

### 4.2 Chunking (`chunk.py`)

- Unit: **words**, not tokens.
- `MAX_WORDS = 100`; `OVERLAP_PERCENT = 0.25`; stride = 75 words.
- `detect_paragraphs()` first tries `r'(\d+\.\s+.*?)(?=\s*\d+\.\s+|$)'` (numbered paragraphs). If no match, it splits on `\n\n`.
- Because `create_cleaned_json()` removed all newlines, the fallback almost never fires; numbered paragraphs are the effective unit. Documents without numbered paragraphs become one giant paragraph and then a sliding window.
- `dynamic_split()` slides a 100-word window with 25-word overlap. It can emit a final chunk shorter than 100 words; 686 chunks (45.5%) hit exactly 100, while 84 chunks are ≤10 words.
- Chunk IDs are random UUID4 values, so re-running chunking produces different IDs for identical content. This breaks idempotent upserts and makes index diffs hard.
- `case_number` is extracted per document with three regexes, then copied to every chunk. The regexes match citation strings (e.g. `2011 SCMR 1433`, `49 of 2009`), which explains the `unknown_case` and wrong-case-number observations.
- Each chunk carries document metadata plus `chunk_type`, `paragraph_number`, `chunk_index`, `total_chunks_in_paragraph`, `word_count`, and `document_index`.
- Chunks are not explicitly prevented from crossing document boundaries; in practice the loop is per document, so they do not. They can cross paragraph boundaries only through the word-window logic.

### 4.3 Embedding and indexing (`embadding.py`)

- Dense model: `BAAI/bge-m3`; dimension 1024; cosine; `normalize_embeddings=True`.
- `MAX_SEQ_LENGTH = 512` is set for indexing, while the query-time model in the app is loaded without this cap (default 8192). For short queries the mismatch is immaterial; for long queries it is a reproducibility inconsistency.
- `BATCH_SIZE = 4` for both the outer loop and `model.encode`.
- Sparse: `TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=2, sublinear_tf=True)` fit on all chunk texts. The same pickled vectorizer is used at query time.
- A **document-level** point is built for each `file_name` by joining all of its chunk texts; 74 such points are embedded and stored. The full text is then truncated by the 512-token model limit. This is not a meaningful full-document embedding.
- Every point gets a fresh `uuid.uuid4()` ID, including document points, so re-indexing creates duplicates rather than replacing old points.
- The collection is created only if absent; there is no schema migration, no payload index creation, and no delete/replace-by-key policy.
- There is no check that the collection's vector dimension matches the model before upsert (the code does use `embed_dim`, which happens to match).
- Sparse vectors are stored as TF-IDF weights, not BM25. No sparse index is configured in Qdrant.

### 4.4 Retrieval (`main.py`, `legal_rag_system.py`)

Both applications implement nearly identical retrieval.

`hybrid_search(query, limit=50)`:

1. Encode the query with BGE-M3, normalized.
2. Transform the query with the TF-IDF vectorizer.
3. Issue a Qdrant `query_points` request with two prefetches:
   - dense, limit 50
   - sparse, limit 50
4. Fuse with `Fusion.RRF`, return up to 50 points.

`aggregate_by_document()` groups payloads by `file_name`. Because document-level points share the same `file_name`, they are merged with chunks.

`rerank_documents()`:

- For **every** chunk payload of every candidate document, builds a `[query, chunk_text]` pair.
- Runs the cross-encoder in one `predict()` call.
- Converts logits to probabilities with a sigmoid.
- Takes the **maximum** probability over the document's chunks as the document score.
- Sorts documents by that score.

`main.py` then returns up to `TOP_K = 3` documents with score ≥ `THRESHOLD = 0.75`, loading full text from `FYP1_extracted_text/<file_name>`. Optional date-range filtering parses four year-range patterns and filters by the first 4-digit sequence in the **filename**; documents without a year are kept.

`legal_rag_system.py` returns only the single best document with score ≥ `THRESHOLD = 0.60`.

Not present: BM25, Qdrant sparse index tuning, query rewriting/expansion, score normalization before fusion, metadata filters other than main's filename-year heuristic, deduplication of retrieved chunks, document-level candidate caps, or per-query top-k control in `legal_rag_system.py`.

### 4.5 Generation and answer construction

Both applications call OpenRouter with `mistralai/mixtral-8x7b-instruct`, `temperature=0.2`.

- `/api/summarize`: prompt uses `text[:3000]`; `max_tokens=250`.
- `/api/chat`: prompt uses `request.context[:3000]`; `max_tokens=200`; explicit refusal strings for out-of-scope questions.
- The prompt tells the model to use only the provided judgment and not to add external law. It does **not** produce structured citations beyond the already-known file name.
- The full retrieved judgment (not selected relevant chunks) is passed to the client, which then truncates to the first 3,000 characters on the server. For long judgments this can omit the actual holding.
- There is no context-window budgeting by tokens, no duplicate-context removal, and no conflict-handling logic.

### 4.6 API and persistence

- FastAPI with CORS `allow_origins=["*"]`, HTTP Bearer sessions.
- `main.py` persists users/sessions/history/saved judgments as JSON under the hard-coded Windows paths.
- `legal_rag_system.py` keeps them in memory (lost on restart).
- Passwords are unsalted SHA-256.
- Search history stores query, result file(s), and score; saved judgments store the full text.
- `/api/upload` accepts PDF/DOCX/TXT, embeds only the first 1,000 characters, stores the full text in the payload, and upserts directly into the production collection. Because retrieval later looks for the filename in `FYP1_extracted_text/`, uploaded documents cannot be returned by `load_full_document()`.

---

## 5. Numerical parameter inventory

The full table is in [`NUMERICAL_CONFIGURATION.md`](NUMERICAL_CONFIGURATION.md). High-level summary:

| Parameter | Value | Location |
|---|---:|---|
| Chunk size | 100 words | `chunk.py:12` |
| Chunk overlap | 25% / 75-word stride | `chunk.py:13` |
| Indexing max sequence length | 512 tokens | `embadding.py:38` |
| Embedding batch size | 4 | `embadding.py:37` |
| Dense dimension | 1024 | model / Qdrant meta |
| TF-IDF features | 15,000 | `embadding.py:82` |
| TF-IDF n-grams | (1, 2) | `embadding.py:83` |
| TF-IDF min_df | 2 | `embadding.py:84` |
| Hybrid prefetch limit per retriever | 50 | `main.py:208`, `legal_rag_system.py:177` |
| Fusion | RRF | `main.py:227`, `legal_rag_system.py:196` |
| Reranker max length | 512 | startup / `main.py:146` |
| Final top-k | 3 (`main`) / 1 (`legal`) | `main.py:45`, `legal_rag_system.py:44` |
| Score threshold | 0.75 (`main`) / 0.60 (`legal`) | `main.py:46`, `legal_rag_system.py:45` |
| LLM temperature | 0.2 | both |
| LLM max tokens (summary/chat) | 250 / 200 | both |
| LLM context characters | 3,000 | both |
| Upload embedding characters | 1,000 | both |
| History returned | last 50 | both |
| Session lifetime | permanent (`main`) / 24 h (`legal`) | both |

---

## 6. Consistency and staleness assessment

- The collection has 1,580 points matching the expected 1,506 chunks + 74 document points.
- Payload `file_name` values all map to the 74 text files in `FYP1_extracted_text/`.
- Dense dimension (1024) matches BGE-M3.
- Sparse dimension is not stored in the collection schema; the pickled vectorizer has 15,000 features, consistent with the indexing code.
- **Staleness risk is high going forward** because point IDs are random UUIDs, there is no content hash, no source-document fingerprint in payloads, no collection version, and no delete-by-source step. Re-running `embadding.py` would append duplicates.
- `legal_rag_system.py` and `main.py` have diverged in thresholds, result cardinality, persistence, and paths. There is no single source of truth.
- The query-time embedding model is not capped to the indexing `max_seq_length`.

---

## 7. Existing tests/evaluation

- No `tests/` directory, no `pytest.ini`, no `pyproject.toml`, no CI config.
- No query dataset, qrels, evaluation script, metric code, or results.
- No README or technical documentation.
- No Git history.

Therefore the true baseline quality is **unknown** at the start of this work. It must be measured with a fixed query set before any optimization.

---

## 8. Baseline constraints and safe execution plan

1. **Do not modify** `judgments for FYP1/`, `FYP1_extracted_text/`, the structured JSON files, `sparse_vectorizer_pak.pkl`, or the existing `pak_legal_rag_multi` collection.
2. Replicate the original retrieval path in a new evaluation module, using the existing collection and vectorizer. No reindexing is needed for the first baseline.
3. Use `main.py`'s behavior as the primary application baseline (top 3, threshold 0.75) because it is the newer, multi-result variant. Also record `legal_rag_system.py`'s single-result threshold 0.60 behavior as a secondary baseline.
4. Evaluate ranking quality with a document-level cutoff of 10, independent of the app's top-3 limit, so Recall@10 and nDCG@10 are meaningful. Report application-mode metrics separately.
5. Build a provisional, manually reviewed query set from the actual judgments; mark it explicitly as provisional and require the FYP student to verify labels.
6. If a chunking experiment is run, create **new** versioned collections (e.g. `pak_legal_rag_c180_o20`) and a new vectorizer file. Never delete or overwrite the current collection.
7. Keep all raw results under `evaluation/results/` with timestamps, config, dataset fingerprint, and model identifiers.

---

## 9. Key risks not addressed by this audit

- The hard-coded OpenRouter key should be treated as compromised and rotated; it is not reproduced in this report.
- Password storage and the `allow_origins=["*"]` CORS policy are not production-safe.
- The corpus of 74 documents is too small and too narrowly focused (mostly bail and criminal appeals) to support broad claims about legal retrieval performance.
- Relevance labels created during this project are provisional unless the FYP student reviews them against the full judgments.
