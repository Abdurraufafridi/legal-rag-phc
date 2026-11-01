# Numerical Configuration Inventory

All values below were read from the current source. “Current” means the value found during the audit on 2026-10-27. Values are not assumed to be correct merely because they are present.

| # | Parameter | Current value | Unit | Location | Used by | Rationale (as found) | Quality impact | Latency / memory impact | Configurable? | Needs tuning? |
|---:|---|---:|---|---|---|---|---|---|---|---|
| 1 | Chunk window | 100 | words | `chunk.py:12` (`MAX_WORDS`) | chunking | Chosen by project author; no token-based rationale in code | Smaller windows increase coverage but can fragment legal reasoning; larger windows improve context but reduce granularity | Larger values reduce point count and embedding/rerank cost | Yes, by editing constant | Yes |
| 2 | Chunk overlap | 25 | percent of window | `chunk.py:13` (`OVERLAP_PERCENT`) | chunking | Author choice | 25% overlap preserves boundary context; higher overlap raises duplicate content and index size | Higher overlap increases chunks, embedding cost, and storage | Yes | Yes |
| 3 | Chunk stride | 75 | words | Derived: `100 - int(100*0.25)` | chunking | Derived from #1–#2 | Controls redundancy | See above | Indirect | With #1/#2 |
| 4 | Chunk minimum | none | words | `chunk.py:51-74` | chunking | No minimum enforced; final window may be 1 word | Very short chunks (84 ≤10 words) add noise and weak embeddings | Slight index bloat | Should be | Yes |
| 5 | Chunk count | 1,506 | points | `structure_dynamic_chunked.json` | indexing | Result of #1–#4 | Determines candidate granularity | Linear cost in embedding, storage, retrieval | N/A | N/A |
| 6 | Document-level points | 74 | points | `embadding.py:151-173` | indexing | “Multi-granularity” design | Duplicates chunk search space; document representations are truncated | Adds 74 vectors and makes every document appear in hybrid candidates | Should be | Yes |
| 7 | Embedding model | `BAAI/bge-m3` | model ID | `embadding.py:30`, `main.py:39`, `legal_rag_system.py:39` | indexing, query | Strong multilingual 1024-d model; already cached | Strong dense quality; may be heavier than needed on CPU | CPU encode ~1.5 s first short query; large model in RAM | Env-configurable recommended | Benchmark alternatives only if quality issue |
| 8 | Embedding dimension | 1024 | floats | model / Qdrant meta | indexing, query | BGE-M3 output | Must equal collection size | 1024-d vectors | No (model-coupled) | No |
| 9 | Dense distance | Cosine | metric | Qdrant collection meta | retrieval | Appropriate for normalized embeddings | Ranking behavior | None | No | No |
| 10 | Dense normalization | `True` | boolean | `embadding.py:212`, query encode calls | indexing, query | Required for cosine consistency | Wrong setting would distort similarity | None | Code constant | No |
| 11 | Indexing max sequence length | 512 | tokens | `embadding.py:38` (`MAX_SEQ_LENGTH`) | indexing | “Prevent attention explosion” | Truncates long chunks/documents; current chunks rarely exceed | Caps CPU cost | Should be | Yes, but chunks are small |
| 12 | Query-time max sequence length | model default 8192 | tokens | model load in API startup | query | Not explicitly set | Index/query preprocessing mismatch for long queries | Higher potential cost on long queries | Should be | No for short queries |
| 13 | Embedding batch size | 4 | samples | `embadding.py:37` | indexing | Conservative CPU setting | None on quality | Lower throughput; low memory | Yes | Yes (throughput) |
| 14 | TF-IDF max features | 15,000 | features | `embadding.py:82` | sparse indexing/query | Author choice | Vocabulary cap; likely adequate for 74 docs | 15k-dim sparse vectors | Yes | Maybe |
| 15 | TF-IDF n-gram range | (1, 2) | n/a | `embadding.py:83` | sparse | Captures legal bigrams (e.g. “further inquiry”) | Improves lexical matching | Moderate vocabulary growth | Yes | Maybe |
| 16 | TF-IDF `min_df` | 2 | documents | `embadding.py:84` | sparse | Removes one-off terms | Slightly reduces noise, may drop case-specific terms | Small | Yes | Maybe |
| 17 | TF-IDF `sublinear_tf` | `True` | boolean | `embadding.py:85` | sparse | Dampens repeated terms | Usually beneficial | None | Yes | Low priority |
| 18 | Sparse weighting | TF-IDF | method | `embadding.py:81-88` | sparse | Project uses TF-IDF, not BM25 | Different term-weighting behavior from BM25 | None | Would require reindex | Only with justification |
| 19 | Hybrid prefetch limit (each retriever) | 50 | points | `main.py:208`, `legal_rag_system.py:177` | retrieval | Author choice | Larger pool improves recall but adds rerank cost and noise | Dominant cost driver with reranker | Yes | Yes |
| 20 | Final hybrid limit | 50 | points | `main.py:228`, `legal_rag_system.py:197` | retrieval | Equals prefetch limit | Caps candidate pool before rerank | Linear rerank cost | Yes | Yes |
| 21 | Fusion method | RRF | enum | `main.py:227`, `legal_rag_system.py:196` | retrieval | Qdrant built-in RRF | Rank-based fusion; scale-free but ignores score gaps | Cheap | Code | Compare DBSF if supported |
| 22 | Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | model ID | `main.py:40`, `legal_rag_system.py:40` | retrieval | General MS MARCO reranker; cached | Query-document interaction usually improves top ranks; domain mismatch possible | ~0.01 s for 2 pairs; grows linearly with pairs | Env-configurable recommended | Benchmark with/without |
| 23 | Reranker max length | 512 | tokens | `main.py:146`, `legal_rag_system.py:123` | retrieval | Model limit | Truncates long chunks | Caps cost | Code | No |
| 24 | Rerank candidate cap | none (all chunks of all candidate docs) | chunks | `main.py:264-283`, `legal_rag_system.py:213-232` | retrieval | No cap in original | Long judgments can dominate; costly | Can be hundreds of pairs | Should be | Yes |
| 25 | Final top-k (`main`) | 3 | judgments | `main.py:45` | API | Author choice | Lower k may omit relevant judgments | Less full-text loading/LLM context | Yes | Yes |
| 26 | Final top-k (`legal`) | 1 | judgment | `legal_rag_system.py:44` | API | Single-best UX | Recall of additional relevant judgments is zero | Lowest | Yes | Yes |
| 27 | Score threshold (`main`) | 0.75 | sigmoid probability | `main.py:46` | API | Author choice | High threshold may return no result; low threshold admits noise | Affects empty-result rate | Yes | Yes |
| 28 | Score threshold (`legal`) | 0.60 | sigmoid probability | `legal_rag_system.py:45` | API | Author choice | Same as above | Yes | Yes | Yes |
| 29 | Date-range parser | 4 patterns | regex | `main.py:192-206` | query preprocessing | Extracts `between/from/…-…/to` years | Only filename years used; metadata dates ignored | Negligible | Should be | Yes |
| 30 | Date filter fallback | include if no year | behavior | `main.py:258-260` | retrieval | Avoids dropping undated docs | Can leak documents outside intended range | None | Code | No |
| 31 | Full text load source | `.txt` only | path | `main.py:285-292`, `legal_rag_system.py:234-241` | retrieval | Original extracted text | Uploaded docs cannot be loaded | Disk read | Should fall back to payload | Yes |
| 32 | LLM model | `mistralai/mixtral-8x7b-instruct` | model ID | both summarization/chat endpoints | generation | OpenRouter choice | Quality/latency depend on provider | Network latency dominates | Yes | Yes |
| 33 | LLM temperature | 0.2 | n/a | both | generation | Low creativity | Good for grounded summarization | None | Yes | Maybe |
| 34 | Summary max tokens | 250 | tokens | both | generation | Author choice | Limits detail | Minor | Yes | Maybe |
| 35 | Chat max tokens | 200 | tokens | both | generation | Author choice | Limits answer length | Minor | Yes | Maybe |
| 36 | LLM context characters | 3,000 | characters | both (`text[:3000]`, `context[:3000]`) | generation | Naive truncation | Can omit holding/reasoning in long judgments | Reduces prompt tokens | Should be token-budgeted | Yes |
| 37 | Upload embedding characters | 1,000 | characters | both (`text[:1000]`) | upload indexing | Naive truncation | Uploaded docs are poorly represented | Low | Should change | Yes |
| 38 | Uploaded payload | full text | characters | both | upload indexing | Store full text | Upload not retrievable due to loader path | Storage | Should store safely | Yes |
| 39 | History page | 50 | records | both `/api/history` | API | Author choice | None on retrieval | Small | Yes | No |
| 40 | Session lifetime (`main`) | no expiry | n/a | `main.py:171` | auth | Author choice | Security risk | None | Should be | Yes |
| 41 | Session lifetime (`legal`) | 24 | hours | `legal_rag_system.py:149` | auth | Author choice | Security risk | None | Yes | No |
| 42 | Password hash | SHA-256, unsalted | algorithm | both | auth | Author choice | Security risk | None | Should be | Yes |
| 43 | Qdrant payload indexes | none | n/a | collection | filtering | Not created | Filtering may be slower | None at this scale | Should be | Low priority |
| 44 | Point IDs | random UUID4 | ID | `embadding.py:187`, upload | indexing | Author choice | Re-runs duplicate data | None | Should be deterministic | Yes |
| 45 | OCR | absent | n/a | `pdf_extraction.py` | ingestion | Not implemented | Scanned PDFs would yield empty text | N/A | Add only if corpus demands | No |
| 46 | Duplicate detection | absent | n/a | ingestion/indexing | ingestion | Not implemented | Repeated files/points possible | N/A | Should be | Yes |
| 47 | Evaluation dataset size | 0 (before this work) | queries | n/a | evaluation | None existed | Unknown quality | N/A | Yes | Yes |
| 48 | Random seeds | none set | n/a | all | all | Not used | Evaluation is deterministic except CPU threading/model nondeterminism | N/A | Should record | Yes |

## Notes on metric-sensitive values

- **Chunk size and embedding truncation interact.** 100-word chunks are approximately 130 English words/tokens, comfortably inside BGE-M3’s 512-token indexing cap. The 512 cap primarily affects the 74 document-level points, which concatenate many chunks.
- **Threshold values are not comparable across models.** The 0.60 and 0.75 thresholds are applied to a sigmoid of an MS MARCO cross-encoder logit. They are not calibrated probabilities and should not be reused blindly if the reranker changes.
- **RRF scores are not probabilities.** They must not be compared to the reranker’s sigmoid scores.
- **The 50-point prefetch is not the same as 50 documents.** Up to 50 chunks can belong to far fewer documents; after aggregation, reranking covers every chunk of each candidate document.
