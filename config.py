"""Central configuration for the legal RAG system.

Defaults match the existing macOS installation. Every path can be overridden
with an environment variable so the application is not tied to one machine.

This module does not read or modify legal documents; it only resolves paths.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is in requirements.txt, but keep import-safe
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default


# ---------------------------------------------------------------------------
# Data locations (read-only originals)
# ---------------------------------------------------------------------------
PDF_DIRECTORY = _path("LEGAL_RAG_PDF_DIR", PROJECT_ROOT / "PHC_Criminal_Judgmentss")
TEXT_DIRECTORY = _path("LEGAL_RAG_TEXT_DIR", PROJECT_ROOT / "PHC_Criminal_Judgmentss_text")
STRUCTURED_JSON = _path("LEGAL_RAG_STRUCTURED_JSON", PROJECT_ROOT / "PHC_Criminal_Judgments_structured.json")
CLEANED_JSON = _path("LEGAL_RAG_CLEANED_JSON", PROJECT_ROOT / "PHC_Criminal_Judgments_cleaned.json")
CHUNKED_JSON = _path("LEGAL_RAG_CHUNKED_JSON", PROJECT_ROOT / "PHC_Criminal_Judgments_chunked.json")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
QDRANT_PATH = _path("LEGAL_RAG_QDRANT_PATH", PROJECT_ROOT / "qdrant_db_phc")
# When set, connect to a Qdrant server instead of embedded local mode.
# Embedded mode is brute-force and becomes slow beyond ~10k points.
QDRANT_URL = os.getenv("LEGAL_RAG_QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("LEGAL_RAG_QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("LEGAL_RAG_COLLECTION", "pak_legal_rag_phc")
VECTORIZER_PATH = _path("LEGAL_RAG_VECTORIZER", PROJECT_ROOT / "sparse_vectorizer_phc.pkl")

# Ingestion defaults
CHUNK_MAX_WORDS = int(os.getenv("LEGAL_RAG_CHUNK_WORDS", "300"))
CHUNK_OVERLAP_PERCENT = float(os.getenv("LEGAL_RAG_CHUNK_OVERLAP", "0.10"))
EMBEDDING_BATCH_SIZE = int(os.getenv("LEGAL_RAG_EMBED_BATCH", "32"))
EMBED_DOCUMENTS = os.getenv("LEGAL_RAG_EMBED_DOCUMENTS", "false").lower() in {"1", "true", "yes", "on"}
RECREATE_COLLECTION = os.getenv(
    "LEGAL_RAG_RECREATE_COLLECTION", "false"
).lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# BGE-M3 is accurate but too slow for CPU-only batch ingestion of this corpus.
EMBEDDING_MODEL_NAME = os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CROSS_ENCODER_NAME = os.getenv(
    "LEGAL_RAG_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("LEGAL_RAG_EMBEDDING_MAX_SEQ", "512"))
RERANKER_MAX_LENGTH = int(os.getenv("LEGAL_RAG_RERANKER_MAX_LENGTH", "512"))

# Torch device for request-path models. CPU is faster than MPS for this
# workload on Intel Macs, so it is the safe default.
MODEL_DEVICE = os.getenv("LEGAL_RAG_MODEL_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Retrieval defaults (main.py behavior)
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("LEGAL_RAG_TOP_K", "3"))
THRESHOLD = float(os.getenv("LEGAL_RAG_THRESHOLD", "0.75"))
PREFETCH_LIMIT = int(os.getenv("LEGAL_RAG_PREFETCH_LIMIT", "50"))
# Number of chunk candidates sent to the cross-encoder. 20 was the best
# latency/quality trade-off in the Phase 1 evaluation.
CANDIDATE_LIMIT = int(os.getenv("LEGAL_RAG_CANDIDATE_LIMIT", "10"))
# Reranking document-level points is the dominant latency cost and is not
# needed for corpus documents. Uploaded files are still included.
RERANK_INCLUDE_DOCUMENT_LEVEL = os.getenv(
    "LEGAL_RAG_RERANK_INCLUDE_DOCUMENT_LEVEL", "false"
).lower() in {"1", "true", "yes", "on"}
RERANK_BATCH_SIZE = int(os.getenv("LEGAL_RAG_RERANK_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Generation (optional; never required for retrieval evaluation)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# LLM backend: "auto" uses local Ollama when reachable, otherwise OpenRouter.
LLM_PROVIDER = os.getenv("LEGAL_RAG_LLM_PROVIDER", "auto")  # auto | ollama | openrouter

# Local Ollama backend
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")

# OpenRouter backend (mistralai/mixtral-8x7b-instruct is no longer served)
LLM_MODEL = os.getenv("LEGAL_RAG_LLM_MODEL", "poolside/laguna-s-2.1:free")
LLM_TEMPERATURE = float(os.getenv("LEGAL_RAG_LLM_TEMPERATURE", "0.2"))
LLM_SUMMARY_MAX_TOKENS = int(os.getenv("LEGAL_RAG_LLM_SUMMARY_MAX_TOKENS", "2500"))
LLM_CHAT_MAX_TOKENS = int(os.getenv("LEGAL_RAG_LLM_CHAT_MAX_TOKENS", "1800"))
LLM_CONTEXT_CHAR_LIMIT = int(os.getenv("LEGAL_RAG_LLM_CONTEXT_CHARS", "3000"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LEGAL_RAG_LLM_TIMEOUT", "90"))
SUMMARY_CACHE_SIZE = int(os.getenv("LEGAL_RAG_SUMMARY_CACHE_SIZE", "128"))

# ---------------------------------------------------------------------------
# Persistence (application; not used by evaluation)
# ---------------------------------------------------------------------------
USERS_DB_FILE = _path("LEGAL_RAG_USERS_DB", PROJECT_ROOT / "users_db.json")
SESSIONS_DB_FILE = _path("LEGAL_RAG_SESSIONS_DB", PROJECT_ROOT / "sessions_db.json")
CHAT_HISTORY_DB_FILE = _path("LEGAL_RAG_CHAT_HISTORY_DB", PROJECT_ROOT / "chat_history_db.json")
CHAT_SESSIONS_DB_FILE = _path("LEGAL_RAG_CHAT_SESSIONS_DB", PROJECT_ROOT / "chat_sessions_db.json")
SAVED_JUDGMENTS_DB_FILE = _path("LEGAL_RAG_SAVED_DB", PROJECT_ROOT / "saved_judgments_db.json")
UPLOADED_DOCS_DB_FILE = _path("LEGAL_RAG_UPLOADED_DOCS_DB", PROJECT_ROOT / "uploaded_documents_db.json")
