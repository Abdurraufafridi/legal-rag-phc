# Actual default retrieval configuration in the application

This documents the exact defaults found in the codebase (as-is). Do **not** change these without approval.

## main.py (primary app)
**File:** `main.py`

| Setting | Value | Location | Notes |
|---------|-------|----------|-------|
| Top-K results | `3` | `TOP_K = 3` (line ~45) | Also overridable via `SearchRequest.top_k` |
| Score threshold | `0.75` | `THRESHOLD = 0.75` (line ~46) | Minimum reranker score to return |
| Embedding model | `BAAI/bge-m3` | `EMBEDDING_MODEL = "BAAI/bge-m3"` (line ~39) | Loaded at startup |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"` (line ~40) | Loaded at startup |
| Hybrid prefetch (dense + sparse) | `50` each | `hybrid_search()` Prefetch dense/sparse limit=50 (lines ~208-227) | Fusion `Fusion.RRF` via `FusionQuery(fusion=Fusion.RRF)` (line ~227) |
| Aggregation | Document-level aggregation enabled | `aggregate_by_document(results)` (line ~301) | Merges chunks; document-level points aggregated by file_name |
| Reranking | Enabled for all candidate chunks | `rerank_documents(query, documents_dict)` (line ~308) | Takes max score per document across all chunks (lines ~264-283) |
| Full-text loading | From extracted files only | `load_full_document(file_name)` reads from `TEXT_DIR / file_name` (line ~285-292) | Uploaded docs not retrievable via this path |

## legal_rag_system.py (alternative app)
**File:** `legal_rag_system.py`

| Setting | Value | Location | Notes |
|---------|-------|----------|-------|
| Top-K | `5` | `TOP_K = 5` (line ~44) | Various flows |
| Threshold | `0.60` | `THRESHOLD = 0.60` (line ~45) | Lower than main.py |
| Hybrid prefetch | `50` each, RRF | Prefetch 50/50, `Fusion.RRF` (lines ~177-196) | Same fusion strategy |
| Reranking | Enabled, all chunks max-aggregated | `rerank_documents()` (lines ~213-232) | Similar logic |

## rag_core.py (evaluation/reusable core)
**File:** `rag_core.py`

| Setting | Value | Location | Notes |
|---------|-------|----------|-------|
| Defaults | Configurable via `RetrievalConfig` | `RetrievalConfig` | Defaults match baseline_main: hybrid RRF, prefetch 50, candidate 50, rerank true, include_document_level true, final_top_k 3, threshold 0.75 |
| Fusion | RRF/DBSF selectable | `RetrievalConfig.fusion` | Qdrant `Fusion.RRF` or `Fusion.DBSF` |

**Summary:** Both apps use hybrid dense+sparse with **RRF fusion**, prefetch 50 each, cross-encoder reranking over all chunks per candidate document (max aggregation). `main.py` is the primary app with defaults TOP_K=3, THRESHOLD=0.75.
