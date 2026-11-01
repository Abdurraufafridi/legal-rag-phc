# Final Report — AI Lawyer Assistant FYP (Resumed & Completed)

**Project root:** `/Users/mc/Desktop/FYP code/rag_env`  
**Report date:** 2026-10-27  
**Status:** Resumed from interrupted state; evaluation completed, documentation produced

---

## 1. Work completed

### 1.1 Audit & documentation (from previous session, verified present)
- [x] **Complete workflow analysis** documented in `docs/AUDIT_REPORT.md` (ingestion, extraction, chunking, embeddings, Qdrant, retrieval, reranking, generation, APIs)
- [x] **Numerical parameter audit** in `docs/NUMERICAL_CONFIGURATION.md` (all parameters inventoried with locations/impacts)

### 1.2 Evaluation infrastructure (created/verified)
- [x] **Retrieval core** (`rag_core.py`): `LegalRetriever`, `RetrievalConfig` (validated), aggregation, ranking with reranker/max-chunk controls, hybrid search mirroring app behavior
- [x] **Dataset loader** (`evaluation/dataset.py`): loads queries/qrels, validates grades, handles out-of-domain separation
- [x] **Metrics** (`evaluation/metrics.py`): Recall@k, Precision@k, MRR@k, MAP@k, nDCG@k with proper handling of empty results
- [x] **Experiment runner** (`evaluation/run_evaluation.py`): reproducible runs with timing, env metadata, dataset fingerprints, writes results JSON + appends to comparison CSV
- [x] **Evaluation dataset**: `evaluation/queries/evaluation_queries.jsonl` (44 queries), `evaluation/queries/relevance_judgments.jsonl` (107 judgments), 4 out-of-domain queries
- [x] **Tests** (`tests/`): 23 unit tests all passing (dataset, metrics, rag_core)

### 1.3 Baseline & controlled experiments (measured)
- [x] **Baseline measured** (replicating `main.py`): `baseline_main` - hybrid RRF, prefetch 50, rerank all chunks, include document-level, top-3, threshold 0.75
- [x] **Experiments completed:** exp01-exp09 (from prior runs), exp10, exp11 (completed on resume)
- [x] **All results persisted** as JSON in `evaluation/results/` with full metrics, latency, rankings, and metadata

### 1.4 Regression testing & comparison
- [x] **Experiment comparison CSV** updated with all experiments: `evaluation/results/experiment_comparison.csv` (13 rows including header)

### 1.5 Comprehensive technical documentation (created on resume)
- [x] **Technical documentation**: `docs/TECHNICAL_DOCUMENTATION.md` - complete workflow from ingestion through evaluation, architecture, experiments, limitations, reproducibility

---

## 2. Work remaining

- **Relevance label review:** The 107 qrels for 44 queries are provisional (created during evaluation setup). FYP supervisor/student should review against full judgments for correctness.
- **Security hardening:** Hard-coded OpenRouter API keys, `allow_origins=["*"]`, unsalted SHA-256 passwords, path portability issues in `main.py` - identified but not modified (out of scope for retrieval optimization without architectural redesign).
- **Upload retrieval bug:** Uploaded documents cannot be loaded by `load_full_document()` (looks only in `FYP1_extracted_text/`). Documented; fix requires loader changes.
- **Chunking improvements:** Word-based chunking with newline-stripped data causes issues (many short chunks, unknown case numbers). Potential improvement but would require reindexing - changes collection; done only if justified with experiments (not necessary for current evaluation goals).
- **Production deployment readiness:** Beyond scope of "analyze, audit, baseline, optimize via controlled experiments, evaluate, document".

---

## 3. Files created or modified

### Created (this resume session)
- `docs/TECHNICAL_DOCUMENTATION.md` (comprehensive workflow documentation)
- `docs/FINAL_REPORT.md` (this report)
- `evaluation/results/exp10_hybrid_dbsf_rerank_20261027T092937Z.json`
- `evaluation/results/exp11_hybrid_rrf_no_rerank_20261027T093135Z.json`

### Modified (this resume session)
- `evaluation/results/experiment_comparison.csv` (appended exp10, exp11 rows)

### Pre-existing (verified, not modified)
- `docs/AUDIT_REPORT.md`, `docs/NUMERICAL_CONFIGURATION.md`
- All evaluation framework files (`rag_core.py`, `evaluation/*`), configs, queries/qrels, existing result JSONs, tests
- All source data, Qdrant DB, models, application code

---

## 4. Baseline vs final evaluation results (actual measurements)

**Evaluation set:** 44 in-domain queries, 107 relevance judgments. All measurements from actual runs.

### Baseline (replicates main.py behavior)
| Metric | Baseline (baseline_main) |
|--------|---------------------------|
| Recall@5 / @10 | 0.8308 / 0.9450 |
| Precision@5 | 0.4050 |
| MRR@10 / MAP@10 | 0.9875 / 0.8371 |
| nDCG@5 / @10 | 0.8960 / 0.9339 |
| Production Recall@3 / Precision@3 | 0.6879 / 0.5083 |
| Production nDCG@3 | 0.8673 |
| Median / p95 total latency | 6.904s / 7.147s |

### Key experimental results (actual)
| Config | Description | Recall@10 | nDCG@10 | Prod R@3 | Prod nDCG@3 | Med lat (s) |
|--------|-------------|-----------|---------|----------|-------------|-------------|
| exp01_chunk_level_only | No document-level points | 0.9408 | 0.9149 | 0.6817 | 0.8515 | 2.321 |
| exp05_chunk_level_cand20 | Chunk-level, cand/prefetch=20 | 0.8879 | 0.9002 | **0.6942** | 0.8579 | **1.758** |
| exp07_chunk_level_tinybert_l2 | Faster reranker | 0.9088 | 0.8994 | 0.6613 | 0.7995 | **0.942** |
| exp08_dense_only_rerank | Dense only | **0.9458** | 0.9197 | 0.6817 | 0.8515 | 2.804 |
| exp10_hybrid_dbsf_rerank | DBSF fusion | 0.9367 | 0.9113 | 0.6817 | 0.8515 | 1.980 |
| exp11_hybrid_rrf_no_rerank | No reranking | 0.9229 | 0.9042 | 0.3113 | 0.4128 | **0.447** |

**Conclusion:** Reranking is essential for production quality. Removing document-level points yields ~3x speedup with minimal quality loss. Reducing candidate pool to 20 gives best production recall@3 in measured runs with further speedup.

---

## 5. Tests executed and actual outcomes

Command executed:
```bash
/Users/mc/Desktop/FYP\ code/rag_env/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

**Result:** Ran 23 tests in 0.006s — **OK**

| Module | Tests | Result |
|--------|-------|--------|
| test_dataset.DatasetTests | 5 | OK |
| test_metrics.QueryMetricTests | 7 | OK |
| test_rag_core.AggregationTests | 4 | OK |
| test_rag_core.ConfigTests | 2 | OK |
| test_rag_core.RankingTests | 5 | OK |
| **Total** | **23** | **All passed** |

---

## 6. Known limitations

- Relevance judgments (107) for 44 queries are provisional and should be reviewed by the FYP student/supervisor
- Corpus is small (74 documents) - results may not generalize
- Document-level points in baseline add cost with limited benefit; upload indexing has retrievability issue
- Chunking is word-based with newline-stripped preprocessing producing some low-quality chunks
- Generation uses naive 3000-char truncation; reranking cost dominates latency in full-chunk mode
- Security issues present in application code (hard-coded keys, insecure auth/CORS) - not addressed
- `main.py` has Windows hard-coded paths (not portable as-is)

---

## 7. Paths to evaluation reports and technical documentation

| Artifact | Path |
|----------|------|
| **Comprehensive technical documentation** | `/Users/mc/Desktop/FYP code/rag_env/docs/TECHNICAL_DOCUMENTATION.md` |
| **Audit report** | `/Users/mc/Desktop/FYP code/rag_env/docs/AUDIT_REPORT.md` |
| **Numerical configuration inventory** | `/Users/mc/Desktop/FYP code/rag_env/docs/NUMERICAL_CONFIGURATION.md` |
| **Final report (this document)** | `/Users/mc/Desktop/FYP code/rag_env/docs/FINAL_REPORT.md` |
| **Experiment comparison (CSV)** | `/Users/mc/Desktop/FYP code/rag_env/evaluation/results/experiment_comparison.csv` |
| **Baseline results (JSON)** | `/Users/mc/Desktop/FYP code/rag_env/evaluation/results/baseline_results.json` |
| **All experiment results (JSON)** | `/Users/mc/Desktop/FYP code/rag_env/evaluation/results/exp01_*.json` through `exp11_*.json` (11 experiment files + baseline) |
| **Evaluation dataset** | `/Users/mc/Desktop/FYP code/rag_env/evaluation/queries/evaluation_queries.jsonl`, `relevance_judgments.jsonl` |
| **Evaluation framework** | `/Users/mc/Desktop/FYP code/rag_env/rag_core.py`, `/Users/mc/Desktop/FYP code/rag_env/evaluation/` |
| **Tests** | `/Users/mc/Desktop/FYP code/rag_env/tests/` (23 tests, all passing) |

---

**Summary:** Resumed from interrupted state, completed missing experiments (exp10, exp11), updated comparison CSV, created comprehensive technical documentation and final report. All measurements are actual; no results fabricated.
