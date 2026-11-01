<div align="center">

# PHC Legal RAG

**Retrieval-augmented legal research over Pakistani criminal judgments.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.17-DC244C)](https://qdrant.tech/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-BGE--small-FFD21E)](https://www.sbert.net/)

[Overview](#overview) · [Data source](#data-source) · [Quick start](#quick-start) · [Evaluation](#evaluation) · [Documentation](#documentation)

</div>

## Overview

PHC Legal RAG is an end-to-end legal research assistant built over **Peshawar High Court criminal judgments**. It combines dense semantic search, sparse keyword search, reciprocal rank fusion, cross-encoder reranking, and document-grounded LLM generation.

Users can search for relevant judgments, inspect the retrieved text, generate summaries, ask follow-up questions about a selected judgment, upload their own documents, and save judgments for later. Retrieval quality is evaluated separately from answer-generation quality.

> The system is a research prototype, not a substitute for professional legal advice.

## Highlights

- **Hybrid retrieval** over dense and sparse representations using Qdrant RRF.
- **Cross-encoder reranking** for more relevant top results.
- **Document-grounded summaries and chat** through OpenRouter or a local Ollama model.
- **Document upload and search** for PDF, DOCX, and TXT files.
- **Reproducible evaluation** with queries, relevance judgments, experiment configs, and saved results.
- **Environment-based configuration** with API keys kept out of source control.

## Data source

The reference corpus was collected from the **[Peshawar High Court Reported Judgments portal](https://www.peshawarhighcourt.gov.pk/PHCCMS/reportedJudgments.php?action=search)**, the official PHC source for reported criminal and civil judgments.

Reference corpus snapshot:

| Item | Count |
|---|---:|
| PDF judgments collected | 2,075 |
| Successfully extracted judgments | 2,053 |
| Searchable text chunks | 20,106 |

The repository **does not redistribute** the judgment PDFs, extracted text, chunk files, vectorizer, or Qdrant index. Add your own corpus to `PHC_Criminal_Judgmentss/` and rebuild the artifacts locally. The judgments remain the property of the Peshawar High Court and are subject to its terms of use.

## How it works

```text
PDF judgments
    -> text extraction and OCR fallback
    -> cleaning and metadata extraction
    -> overlapping chunks
    -> dense embeddings (BGE-small) + sparse TF-IDF
    -> Qdrant hybrid index
    -> RRF fusion and cross-encoder reranking
    -> grounded summaries and document Q&A
```

**Stack:** FastAPI, Uvicorn, Qdrant, BGE-small-en-v1.5, `ms-marco-MiniLM-L-6-v2`, scikit-learn TF-IDF, OpenRouter, and Ollama.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Abdurraufafridi/legal-rag-phc.git
cd legal-rag-phc

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`, or use the local Ollama backend by setting `LEGAL_RAG_LLM_PROVIDER=ollama`. Never commit `.env`.

### 3. Prepare the corpus and index

Place the source PDFs in `PHC_Criminal_Judgmentss/`, start Qdrant on port `6333`, then run:

```bash
python pdf_extraction.py   # PDF -> text + structured/cleaned JSON
python chunk.py            # cleaned JSON -> chunks
python embadding.py        # chunks -> dense + sparse Qdrant index
```

Scanned PDFs use Tesseract OCR, which must be installed separately. The pipeline skips existing extracted text by default.

### 4. Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Open <http://localhost:8001>.

## Evaluation

The `evaluation/` directory contains 44 queries, 107 graded relevance judgments, 11 retrieval configurations, and saved result files.

```bash
python -m evaluation.run_evaluation \
  --config evaluation/configs/baseline_main.json \
  --output evaluation/results/baseline_results.json
```

Results in `evaluation/results/` are retained as research evidence. Some may reflect an earlier corpus snapshot, so check the dataset metadata before comparing them directly.

## Project layout

```text
.
├── main.py                  # FastAPI app, UI, retrieval, and generation
├── config.py                # Environment-based configuration
├── pdf_extraction.py        # PDF extraction and metadata pipeline
├── chunk.py                 # Cleaning and overlapping chunking
├── embadding.py             # Dense/sparse embeddings and Qdrant upsert
├── rag_core.py              # Read-only retrieval core used by evaluation
├── legal_rag_system.py      # Earlier standalone implementation
├── evaluation/              # Queries, metrics, configs, and results
├── tests/                   # Unit tests
└── docs/                    # Technical and audit documentation
```

## Documentation

- [Complete project technical notes](docs/COMPLETE_PROJECT_TECHNICAL_NOTES.md)
- [Technical documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Numerical configuration reference](docs/NUMERICAL_CONFIGURATION.md)
- [Verification report](docs/VERIFICATION_REPORT.md)

## Disclaimer

This project is intended for research and educational use. It is **not legal advice**. Always verify retrieved text, summaries, and answers against the official judgment before relying on them.

<div align="center">

Built by [Abdur Rauf Afridi](https://github.com/Abdurraufafridi)

</div>
