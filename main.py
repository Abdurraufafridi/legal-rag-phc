# ============================================================
# LEGAL RAG SYSTEM — FastAPI + Ollama (Local LLM, GPU-Accelerated)
# Features: Auth, Sessions, History, Saved Judgments, Upload,
#           Hybrid Retrieval (Qdrant), Dense+Sparse Embeddings,
#           CrossEncoder Reranking, Summarization, Chat
# LLM Backend: Ollama (llama3:instruct) via OpenAI-compatible API
# GPU: NVIDIA RTX 3050 — Ollama auto-uses CUDA when available
# ============================================================

import os
import io
import pickle
import json
import uuid
import hashlib
import logging
import traceback
import threading
from datetime import datetime
from typing import Optional
from collections import defaultdict, OrderedDict
import re

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    SparseVector,
    FusionQuery,
    Fusion,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from openai import OpenAI

import PyPDF2
from docx import Document

import config as app_config

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("legal_rag")

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = app_config.EMBEDDING_MODEL_NAME
CROSS_ENCODER_NAME   = app_config.CROSS_ENCODER_NAME

QDRANT_PATH          = str(app_config.QDRANT_PATH)
QDRANT_URL           = app_config.QDRANT_URL
QDRANT_API_KEY       = app_config.QDRANT_API_KEY
COLLECTION_NAME      = app_config.COLLECTION_NAME
VECTORIZER_PATH      = str(app_config.VECTORIZER_PATH)
ORIGINAL_TEXT_FOLDER = str(app_config.TEXT_DIRECTORY)

TOP_K     = app_config.TOP_K
THRESHOLD = app_config.THRESHOLD
PREFETCH_LIMIT = app_config.PREFETCH_LIMIT
CANDIDATE_LIMIT = app_config.CANDIDATE_LIMIT
RERANK_INCLUDE_DOCUMENT_LEVEL = app_config.RERANK_INCLUDE_DOCUMENT_LEVEL
RERANK_BATCH_SIZE = app_config.RERANK_BATCH_SIZE
SUMMARY_CACHE_SIZE = app_config.SUMMARY_CACHE_SIZE
LLM_CONTEXT_CHAR_LIMIT = app_config.LLM_CONTEXT_CHAR_LIMIT
LLM_SUMMARY_MAX_TOKENS = app_config.LLM_SUMMARY_MAX_TOKENS
LLM_CHAT_MAX_TOKENS = app_config.LLM_CHAT_MAX_TOKENS
OPENROUTER_API_KEY = app_config.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = app_config.OPENROUTER_BASE_URL
LLM_PROVIDER = app_config.LLM_PROVIDER
LLM_MODEL = app_config.LLM_MODEL
LLM_TEMPERATURE = app_config.LLM_TEMPERATURE
LLM_TIMEOUT_SECONDS = app_config.LLM_TIMEOUT_SECONDS

_summary_cache = OrderedDict()
_summary_cache_lock = threading.Lock()
_openrouter_client = None

# Ollama local configuration
OLLAMA_BASE_URL = app_config.OLLAMA_BASE_URL
OLLAMA_API_KEY  = app_config.OLLAMA_API_KEY
OLLAMA_MODEL    = app_config.OLLAMA_MODEL

# Persistent storage paths
BASE_DIR              = str(app_config.PROJECT_ROOT)
USERS_DB_FILE         = str(app_config.USERS_DB_FILE)
SESSIONS_DB_FILE      = str(app_config.SESSIONS_DB_FILE)
CHAT_HISTORY_DB_FILE  = str(app_config.CHAT_HISTORY_DB_FILE)
CHAT_SESSIONS_DB_FILE = str(app_config.CHAT_SESSIONS_DB_FILE)
SAVED_JUDGMENTS_DB_FILE = str(app_config.SAVED_JUDGMENTS_DB_FILE)
UPLOADED_DOCS_DB_FILE = str(app_config.UPLOADED_DOCS_DB_FILE)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# ============================================================
# PERSISTENT DATABASE HELPERS
# ============================================================

def load_db(file_path: str, default):
    """Load a JSON database from disk, or return default if absent."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_db(file_path: str, data):
    """Persist a JSON-serialisable object to disk atomically."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# In-memory stores (loaded from disk at startup)
users_db    = load_db(USERS_DB_FILE, {})
sessions_db = load_db(SESSIONS_DB_FILE, {})
chat_history_db     = defaultdict(list, load_db(CHAT_HISTORY_DB_FILE, {}))
chat_sessions_db    = defaultdict(list, load_db(CHAT_SESSIONS_DB_FILE, {}))
saved_judgments_db  = defaultdict(list, load_db(SAVED_JUDGMENTS_DB_FILE, {}))
uploaded_documents_db = load_db(UPLOADED_DOCS_DB_FILE, {})
_chat_lock = threading.Lock()
_upload_lock = threading.Lock()


def _remember_uploaded_document(email: str, file_name: str, text: str) -> None:
    """Keep uploaded text so the retrieval layer can load it after indexing."""
    uploaded_documents_db.setdefault(email, {})[file_name] = {
        "text": text,
        "uploaded_at": datetime.now().isoformat(),
    }


def _get_uploaded_document_text(file_name: str, email: Optional[str] = None) -> Optional[str]:
    """Return the stored text for an uploaded document, if available."""
    candidate_stores = (
        [uploaded_documents_db.get(email, {})]
        if email
        else list(uploaded_documents_db.values())
    )
    for documents in candidate_stores:
        if not isinstance(documents, dict):
            continue
        record = documents.get(file_name)
        if isinstance(record, str) and record.strip():
            return record
        if isinstance(record, dict):
            text = record.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


def _new_chat(title: str = "New chat") -> dict:
    now = datetime.now().isoformat()
    return {
        "id": str(uuid.uuid4()),
        "title": title or "New chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def _find_chat(email: str, chat_id: str) -> Optional[dict]:
    for chat in chat_sessions_db.get(email, []):
        if chat.get("id") == chat_id:
            return chat
    return None


def create_chat(email: str, title: str = "New chat") -> dict:
    chat = _new_chat(title)
    with _chat_lock:
        chat_sessions_db[email].append(chat)
        save_db(CHAT_SESSIONS_DB_FILE, chat_sessions_db)
    return chat


def get_or_create_chat(email: str, chat_id: Optional[str], first_query: str = "") -> dict:
    if chat_id:
        chat = _find_chat(email, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat
    return create_chat(email, first_query[:60] if first_query else "New chat")


def append_chat_message(
    email: str,
    chat: dict,
    role: str,
    content: str,
    judgments: Optional[list] = None,
) -> dict:
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    if judgments:
        message["judgments"] = judgments

    with _chat_lock:
        chat.setdefault("messages", []).append(message)
        if role == "user" and chat.get("title") in (None, "", "New chat"):
            chat["title"] = content[:60]
        chat["updated_at"] = datetime.now().isoformat()
        save_db(CHAT_SESSIONS_DB_FILE, chat_sessions_db)
    return message


def migrate_legacy_history() -> None:
    """Convert old flat search history into one chat per user."""
    legacy = load_db(CHAT_HISTORY_DB_FILE, {})
    changed = False
    for email, items in legacy.items():
        if chat_sessions_db.get(email) or not isinstance(items, list) or not items:
            continue
        messages = []
        for item in items:
            if not isinstance(item, dict) or not item.get("query"):
                continue
            timestamp = item.get("timestamp") or datetime.now().isoformat()
            messages.append({
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": item["query"],
                "timestamp": timestamp,
            })
            count = item.get("result_count", 0) or 0
            files = item.get("result_files") or []
            if count:
                content = f"Found {count} relevant judgment(s)."
                if files:
                    content += "\n" + "\n".join(files)
            else:
                content = "No strong judgment found. Try refining your query."
            messages.append({
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": content,
                "timestamp": timestamp,
            })
        if not messages:
            continue
        timestamps = [m["timestamp"] for m in messages]
        chat_sessions_db[email].append({
            "id": str(uuid.uuid4()),
            "title": "Previous searches",
            "created_at": min(timestamps),
            "updated_at": max(timestamps),
            "messages": messages,
        })
        changed = True
    if changed:
        save_db(CHAT_SESSIONS_DB_FILE, chat_sessions_db)


migrate_legacy_history()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    top_k: Optional[int] = TOP_K

class ChatRequest(BaseModel):
    question: str
    context: str
    session_id: str
    chat_id: Optional[str] = None
    card_id: str

class SaveJudgmentRequest(BaseModel):
    file_name: str
    text: str
    score: float

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Legal RAG System — Ollama Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============================================================
# GLOBAL MODEL OBJECTS (initialised in startup event)
# ============================================================

embedding_model  = None
cross_encoder    = None
sparse_vectorizer = None
qdrant_client   = None
llm_client      = None

@app.on_event("startup")
async def load_models():
    global embedding_model, cross_encoder, sparse_vectorizer, qdrant_client, llm_client

    logger.info("🚀 Starting model loading sequence …")

    # Dense embedding model (BAAI/bge-m3)
    logger.info("  • Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=app_config.MODEL_DEVICE)

    # Cross-encoder reranker
    logger.info("  • Loading cross-encoder: %s", CROSS_ENCODER_NAME)
    cross_encoder = CrossEncoder(
        CROSS_ENCODER_NAME,
        max_length=app_config.RERANKER_MAX_LENGTH,
        device=app_config.MODEL_DEVICE,
    )

    # Sparse vectorizer (TF-IDF / BM25 style)
    logger.info("  • Loading sparse vectorizer from: %s", VECTORIZER_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        sparse_vectorizer = pickle.load(f)

    # Qdrant vector database (local disk)
    if QDRANT_URL:
        logger.info("  • Connecting to Qdrant server at: %s", QDRANT_URL)
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=60,
        )
    else:
        logger.info("  • Connecting to embedded Qdrant at: %s", QDRANT_PATH)
        qdrant_client = QdrantClient(path=QDRANT_PATH)

    # Ollama LLM client (OpenAI-compatible interface)
    # Ollama listens on port 11434 and mirrors the OpenAI /v1 API.
    # GPU acceleration is handled entirely by Ollama — no extra config needed here.
    logger.info("  • Connecting to Ollama at: %s  (model: %s)", OLLAMA_BASE_URL, OLLAMA_MODEL)
    llm_client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
    )

    logger.info("✅ All models loaded and ready.")

# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_session(email: str) -> str:
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "email": email,
        "created_at": datetime.now().isoformat(),
    }
    save_db(SESSIONS_DB_FILE, sessions_db)
    return session_id

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    if token not in sessions_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )
    return sessions_db[token]["email"]

# ============================================================
# SEARCH / RETRIEVAL HELPERS
# ============================================================

def parse_date_range(query: str):
    """Extract a (start_year, end_year) tuple from the query string, or (None, None)."""
    patterns = [
        r'between\s+(\d{4})\s+to\s+(\d{4})',
        r'from\s+(\d{4})\s+to\s+(\d{4})',
        r'(\d{4})\s*-\s*(\d{4})',
        r'(\d{4})\s+to\s+(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None

def _chunk_level_filter(include_document_level: bool = False):
    """Restrict candidate retrieval to chunk-level points unless opted out."""
    if include_document_level:
        return None
    return Filter(
        must=[FieldCondition(key="level", match=MatchValue(value="chunk"))]
    )


def hybrid_search(
    query: str,
    limit: int = PREFETCH_LIMIT,
    include_document_level: bool = RERANK_INCLUDE_DOCUMENT_LEVEL,
):
    """Perform dense + sparse hybrid search via Qdrant RRF fusion."""
    dense_vector = embedding_model.encode(query, normalize_embeddings=True).tolist()

    sparse_query  = sparse_vectorizer.transform([query])
    sparse_vector = SparseVector(
        indices=sparse_query[0].indices.tolist(),
        values=sparse_query[0].data.tolist(),
    )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector,  using="dense",  limit=limit),
            Prefetch(query=sparse_vector, using="sparse", limit=limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=_chunk_level_filter(include_document_level),
        limit=limit,
        with_payload=True,
    )
    return results.points


def aggregate_by_document(points) -> dict:
    """Group retrieved chunks by source document filename."""
    documents = defaultdict(list)
    for p in points:
        file_name = p.payload.get("file_name", "unknown")
        documents[file_name].append(p.payload)
    return documents

def filter_by_date_range(documents_dict: dict, start_year: int, end_year: int) -> dict:
    """Keep only documents whose filename contains a year within [start_year, end_year]."""
    if start_year is None or end_year is None:
        return documents_dict

    filtered = defaultdict(list)
    for file_name, chunks in documents_dict.items():
        year_match = re.search(r'(\d{4})', file_name)
        if year_match:
            year = int(year_match.group(1))
            if start_year <= year <= end_year:
                filtered[file_name] = chunks
        else:
            filtered[file_name] = chunks  # Include if year cannot be parsed
    return filtered

def rerank_documents(query: str, documents_dict: dict) -> list:
    """Score every retrieved chunk in one cross-encoder batch, then aggregate."""
    pairs = []
    pair_files = []
    for file_name, chunks in documents_dict.items():
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            pairs.append([query, text])
            pair_files.append(file_name)

    if not pairs:
        return []

    scores = cross_encoder.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )
    scores = np.asarray(scores, dtype=float).reshape(-1)
    normalized = 1 / (1 + np.exp(-scores))

    doc_scores = {}
    for file_name, score in zip(pair_files, normalized):
        score = float(score)
        if file_name not in doc_scores or score > doc_scores[file_name]:
            doc_scores[file_name] = score

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def load_full_document(file_name: str, email: Optional[str] = None) -> Optional[str]:
    """Load the full text of a judgment from uploaded storage or disk."""
    uploaded_text = _get_uploaded_document_text(file_name, email)
    if uploaded_text:
        return uploaded_text

    file_path = os.path.join(ORIGINAL_TEXT_FOLDER, file_name)
    if not os.path.exists(file_path):
        logger.warning("Full-text file not found: %s", file_path)
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _reconstruct_text_from_chunks(chunks: list) -> Optional[str]:
    """Rebuild document text from retrieved chunks when no stored copy exists."""
    entries = []
    seen = set()
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        key = chunk.get("chunk_id") or text
        if key in seen:
            continue
        seen.add(key)
        metadata = chunk.get("metadata") or {}
        index = metadata.get("chunk_index")
        entries.append((index if isinstance(index, int) else len(entries), text))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in entries)


def _retrieve_once(query: str, top_k: int, email: Optional[str] = None) -> list:
    """Run one hybrid-search → rerank → threshold pass."""
    points = hybrid_search(query)
    if not points:
        return []
    if CANDIDATE_LIMIT:
        points = points[:CANDIDATE_LIMIT]

    documents_dict = aggregate_by_document(points)

    start_year, end_year = parse_date_range(query)
    if start_year and end_year:
        documents_dict = filter_by_date_range(documents_dict, start_year, end_year)

    ranked = rerank_documents(query, documents_dict)

    valid_judgments = []
    for file_name, score in ranked:
        if score >= THRESHOLD and len(valid_judgments) < top_k:
            full_text = load_full_document(file_name, email)
            if not full_text:
                full_text = _reconstruct_text_from_chunks(documents_dict.get(file_name, []))
            if full_text:
                valid_judgments.append({
                    "file_name": file_name,
                    "score": score,
                    "full_text": full_text,
                })

    return valid_judgments


def _query_expansions(query: str) -> list:
    """Expand short keyword queries that the cross-encoder otherwise scores low."""
    q = query.lower()
    homicide = re.search(r"\b(murder|kill|killed|killing|homicide)\b", q)
    property_dispute = re.search(r"\b(property|land|landed|plot)\b", q)
    if not (homicide and property_dispute):
        return []

    normalized = re.sub(r"\b(kill|killed|killing)\b", "murder", query)
    return [
        f"{normalized} land dispute motive enmity",
        f"{normalized} dispute over landed property motive",
    ]


def retrieve_judgments(query: str, top_k: int = TOP_K, email: Optional[str] = None) -> list:
    """End-to-end retrieval with a query-expansion fallback for sparse keywords."""
    results = _retrieve_once(query, top_k, email)
    if results:
        return results

    for expanded_query in _query_expansions(query):
        logger.info("Retrieval fallback | original='%s' | expanded='%s'", query, expanded_query)
        results = _retrieve_once(expanded_query, top_k, email)
        if results:
            return results

    return []

# ============================================================
# FILE PROCESSING HELPERS
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)

def extract_text_from_docx(file_content: bytes) -> str:
    doc = Document(io.BytesIO(file_content))
    return "\n".join(p.text for p in doc.paragraphs)

# ============================================================
# LLM CALL HELPER — centralised Ollama interaction with logging
# ============================================================

def call_llm(messages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call the configured LLM backend (OpenRouter or local Ollama).

    Free OpenRouter models sometimes route to a reasoning model.  If the token
    budget is consumed by internal reasoning, ``message.content`` is empty.  We
    retry once with a larger budget and an instruction to emit only the final
    answer, then fail loudly instead of returning ``None`` to the UI.
    """
    global _openrouter_client

    provider = (LLM_PROVIDER or "auto").lower()
    use_openrouter = provider == "openrouter" or (provider == "auto" and OPENROUTER_API_KEY)
    logger.info(
        "→ LLM request | provider=%s | messages=%d | max_tokens=%d | temp=%.2f",
        provider, len(messages), max_tokens, temperature,
    )

    if use_openrouter and not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    if use_openrouter and _openrouter_client is None:
        _openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    if not use_openrouter and llm_client is None:
        raise RuntimeError("Ollama client is not initialised.")

    def request(msgs: list, tokens: int):
        if use_openrouter:
            return _openrouter_client.chat.completions.create(
                model=LLM_MODEL,
                messages=msgs,
                temperature=temperature,
                max_tokens=tokens,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        return llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=tokens,
        )

    retry_messages = [
        {
            "role": "system",
            "content": "Return only the final answer. Do not include internal reasoning or analysis.",
        },
        *messages,
    ]
    retry_tokens = min(max(max_tokens * 2, 2500), 4096)

    for attempt, (msgs, tokens) in enumerate(
        ((messages, max_tokens), (retry_messages, retry_tokens))
    ):
        response = request(msgs, tokens)
        message = response.choices[0].message
        answer = (message.content or "").strip()
        if answer:
            logger.info("← LLM response | length=%d chars", len(answer))
            return answer

        reasoning = getattr(message, "reasoning", None) or ""
        logger.warning(
            "← LLM returned empty content | attempt=%d | finish_reason=%s | reasoning_chars=%d",
            attempt + 1,
            response.choices[0].finish_reason,
            len(reasoning),
        )

    raise RuntimeError(
        "The language model returned an empty response after a retry. "
        "Please try again or switch the configured model."
    )


def call_ollama(messages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Backward-compatible alias for the local Ollama call."""
    return call_llm(messages, max_tokens=max_tokens, temperature=temperature)


def get_cached_summary(key: str):
    with _summary_cache_lock:
        if key not in _summary_cache:
            return None
        value = _summary_cache.pop(key)
        _summary_cache[key] = value
        return value


def cache_summary(key: str, summary: str) -> None:
    if SUMMARY_CACHE_SIZE <= 0:
        return
    with _summary_cache_lock:
        _summary_cache.pop(key, None)
        _summary_cache[key] = summary
        while len(_summary_cache) > SUMMARY_CACHE_SIZE:
            _summary_cache.popitem(last=False)


def chunk_document_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list:
    """Split arbitrary uploaded text into overlapping word-boundary chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    total_words = len(words)
    while start < total_words:
        end = start
        length = 0
        while end < total_words:
            addition = len(words[end]) + (1 if length else 0)
            if length + addition > max_chars and end > start:
                break
            length += addition
            end += 1

        chunks.append(" ".join(words[start:end]))
        if end >= total_words:
            break

        if overlap <= 0:
            start = end
            continue

        index = end - 1
        overlap_chars = 0
        while index >= start:
            addition = len(words[index]) + (1 if overlap_chars else 0)
            if overlap_chars + addition > overlap:
                break
            overlap_chars += addition
            index -= 1
        next_start = index + 1
        start = max(start + 1, next_start)

    return chunks


def index_uploaded_document(text: str, file_name: str, email: str) -> int:
    """Embed an uploaded document as chunk-level points owned by the user."""
    chunks = chunk_document_text(text, max_chars=1000, overlap=100)
    if not chunks:
        return 0

    dense_vectors = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )
    sparse_vectors = sparse_vectorizer.transform(chunks)

    points = []
    for index, (chunk_text, dense_vector, sparse_row) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        sparse_vector = SparseVector(
            indices=sparse_row.indices.tolist(),
            values=sparse_row.data.tolist(),
        )
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vector.tolist(), "sparse": sparse_vector},
                payload={
                    "file_name": file_name,
                    "text": chunk_text,
                    "chunk_id": f"uploaded_{file_name}_{index}",
                    "level": "chunk",
                    "uploaded_by": email,
                    "metadata": {"file_name": file_name, "chunk_index": index},
                },
            )
        )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    with _upload_lock:
        _remember_uploaded_document(email, file_name, text)
    return len(points)


# ============================================================
# API ENDPOINTS — AUTHENTICATION
# ============================================================

@app.post("/api/register")
async def register(user: UserRegister):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already exists")

    users_db[user.email] = {
        "email": user.email,
        "password": hash_password(user.password),
        "full_name": user.full_name,
        "created_at": datetime.now().isoformat(),
        "is_new_user": True,
    }
    save_db(USERS_DB_FILE, users_db)
    logger.info("New user registered: %s", user.email)
    return {"message": "Registration successful", "email": user.email, "is_new_user": True}

@app.post("/api/login")
async def login(user: UserLogin):
    stored = users_db.get(user.email)
    if not stored or hash_password(user.password) != stored["password"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_id = create_session(user.email)
    is_returning = not stored.get("is_new_user", False)

    if stored.get("is_new_user"):
        stored["is_new_user"] = False
        save_db(USERS_DB_FILE, users_db)

    logger.info("User logged in: %s (returning=%s)", user.email, is_returning)
    return {
        "session_id": session_id,
        "email": user.email,
        "full_name": stored["full_name"],
        "welcome_message": "Welcome Back" if is_returning else "Welcome",
    }

@app.post("/api/logout")
async def logout(email: str = Depends(get_current_user)):
    to_remove = [sid for sid, sess in sessions_db.items() if sess["email"] == email]
    for sid in to_remove:
        del sessions_db[sid]
    save_db(SESSIONS_DB_FILE, sessions_db)
    logger.info("User logged out: %s", email)
    return {"message": "Logged out successfully"}

@app.get("/api/profile")
async def get_profile(email: str = Depends(get_current_user)):
    user = users_db.get(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"email": user["email"], "full_name": user["full_name"], "created_at": user["created_at"]}

# ============================================================
# API ENDPOINTS — SEARCH & RETRIEVAL
# ============================================================

@app.post("/api/search")
async def search_judgment(request: SearchRequest, email: str = Depends(get_current_user)):
    logger.info("Search request | user=%s | query='%s' | top_k=%d", email, request.query, request.top_k or TOP_K)
    try:
        top_k = request.top_k or TOP_K
        chat = get_or_create_chat(email, request.chat_id, request.query)
        append_chat_message(email, chat, "user", request.query)

        try:
            judgments = retrieve_judgments(request.query, top_k, email=email)
        except Exception:
            logger.error("Retrieval failed for user=%s:\n%s", email, traceback.format_exc())
            append_chat_message(
                email,
                chat,
                "assistant",
                "Search failed because the retrieval service is unavailable. Please try again.",
            )
            raise HTTPException(
                status_code=503,
                detail="Search service is unavailable. Make sure the vector database is running and try again.",
            )

        if judgments:
            append_chat_message(
                email,
                chat,
                "assistant",
                f"Found {len(judgments)} relevant judgment(s) with score ≥ 75%",
                judgments=judgments,
            )
            logger.info("Search returned %d judgment(s) for user=%s", len(judgments), email)
            return {
                "success": True,
                "judgments": judgments,
                "chat_id": chat["id"],
                "chat_title": chat["title"],
            }

        append_chat_message(
            email,
            chat,
            "assistant",
            "No strong judgment found. Try refining your query.",
        )
        logger.info("No judgments met threshold for user=%s query='%s'", email, request.query)
        return {
            "success": False,
            "message": "No strong judgment found. Try refining your query.",
            "chat_id": chat["id"],
            "chat_title": chat["title"],
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("Error in /api/search:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Search failed. Check server logs for details.")

# ============================================================
# API ENDPOINTS — SUMMARIZE  (fixed: now uses OLLAMA_MODEL)
# ============================================================

@app.post("/api/summarize")
async def summarize_judgment(request: dict, email: str = Depends(get_current_user)):
    """
    Summarise a retrieved judgment using the local Ollama LLM.

    Previous bug: model was still set to 'mistralai/mixtral-8x7b-instruct'
    (an OpenRouter model name) even after the client was switched to Ollama.
    Ollama cannot locate that model locally → 500 error.
    Fix: use OLLAMA_MODEL = 'llama3:instruct'.
    """
    text = request.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    logger.info("Summarize request | user=%s | text_length=%d chars", email, len(text))

    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = get_cached_summary(cache_key)
    if cached and cached.strip():
        logger.info("Summarize cache hit | user=%s", email)
        return {"summary": cached, "cached": True}

    # Limit context to fit the configured LLM context window
    context_text = text[:LLM_CONTEXT_CHAR_LIMIT]

    prompt = f"""You are a legal assistant. Your task is to summarize the judgment provided below.

STRICT RULES:
1. Use ONLY the information present in the judgment text below. Do NOT add any external legal knowledge.
2. Be specific: mention exact parties, dates, courts, sections, and outcomes found in the text.
3. Do NOT provide general legal explanations or background information.
4. If a detail is not present in the text, do not invent or assume it.
5. Structure your summary as: (a) Parties involved, (b) Key facts, (c) Legal issues, (d) Decision/Outcome.

JUDGMENT TEXT:
{context_text}

Provide a concise, grounded summary based solely on the above text."""

    try:
        summary = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_SUMMARY_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        if not summary or not summary.strip():
            raise RuntimeError("The language model returned an empty summary.")
        cache_summary(cache_key, summary)
        logger.info("Summarize success | user=%s | summary_length=%d chars", email, len(summary))
        return {"summary": summary, "cached": False}

    except Exception:
        logger.error("Error in /api/summarize for user=%s:\n%s", email, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Summarization failed because the language model returned no text. Please try again.",
        )

# ============================================================
# API ENDPOINTS — CHAT  (fixed: now uses OLLAMA_MODEL)
# ============================================================

@app.post("/api/chat")
async def chat_with_judgment(request: ChatRequest, email: str = Depends(get_current_user)):
    """
    Answer a user question grounded in a specific judgment using the local Ollama LLM.

    Previous bug: model was still set to 'mistralai/mixtral-8x7b-instruct'.
    Fix: use OLLAMA_MODEL = 'llama3:instruct'.
    """
    # Handle exit commands silently (no LLM call needed)
    if request.question.lower().strip() in {"exit", "quit", "/exit", "/quit"}:
        logger.info("Chat exit command | user=%s | card=%s", email, request.card_id)
        return {"answer": None, "exit_chat": True, "silent": True, "card_id": request.card_id}

    logger.info(
        "Chat request | user=%s | card=%s | question_length=%d chars",
        email, request.card_id, len(request.question),
    )

    context_text = request.context[:LLM_CONTEXT_CHAR_LIMIT]

    prompt = f"""You are a legal assistant answering questions strictly about the judgment provided below.

STRICT RULES:
1. Answer ONLY using information explicitly present in the judgment text below.
2. Be specific and precise — cite exact facts, dates, parties, sections, or rulings from the text.
3. Do NOT provide general legal knowledge, commentary, or background explanations.
4. If the question cannot be answered from the judgment text, respond with:
   "I cannot answer this question from the provided judgment. Please ask something specific to this judgment."
5. If the question is unrelated to the judgment (e.g. general trivia, other laws, other cases), respond with:
   "This question is outside the scope of the retrieved judgment. Please ask a question related to this specific judgment."

JUDGMENT TEXT:
{context_text}

QUESTION:
{request.question}

Answer strictly from the judgment text above:"""

    try:
        answer = call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_CHAT_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        chat = get_or_create_chat(email, request.chat_id, request.question)
        append_chat_message(email, chat, "user", request.question)
        append_chat_message(email, chat, "assistant", answer)
        logger.info("Chat success | user=%s | card=%s | answer_length=%d chars", email, request.card_id, len(answer))
        return {
            "answer": answer,
            "exit_chat": False,
            "silent": False,
            "card_id": request.card_id,
            "chat_id": chat["id"],
        }

    except Exception:
        logger.error("Error in /api/chat for user=%s card=%s:\n%s", email, request.card_id, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Chat failed because the language model returned no text. Please try again.",
        )

# ============================================================
# API ENDPOINTS — HISTORY & SAVED JUDGMENTS
# ============================================================

@app.get("/api/chats")
async def list_chats(email: str = Depends(get_current_user)):
    chats = sorted(
        (
            chat
            for chat in chat_sessions_db.get(email, [])
            if chat.get("messages")
        ),
        key=lambda chat: chat.get("updated_at", ""),
        reverse=True,
    )
    return {
        "chats": [
            {
                "id": chat["id"],
                "title": chat.get("title") or "New chat",
                "created_at": chat.get("created_at"),
                "updated_at": chat.get("updated_at"),
                "message_count": len(chat.get("messages", [])),
            }
            for chat in chats
        ]
    }


@app.post("/api/chats")
async def start_chat(email: str = Depends(get_current_user)):
    return {"chat": create_chat(email)}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, email: str = Depends(get_current_user)):
    chat = _find_chat(email, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": chat}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, email: str = Depends(get_current_user)):
    with _chat_lock:
        user_chats = chat_sessions_db.get(email, [])
        remaining = [chat for chat in user_chats if chat.get("id") != chat_id]
        if len(remaining) == len(user_chats):
            raise HTTPException(status_code=404, detail="Chat not found")
        chat_sessions_db[email] = remaining
        save_db(CHAT_SESSIONS_DB_FILE, chat_sessions_db)
    logger.info("Chat deleted | user=%s | chat=%s", email, chat_id)
    return {"message": "Chat deleted successfully", "chat_id": chat_id}


@app.get("/api/history")
async def get_history(email: str = Depends(get_current_user)):
    history = chat_history_db.get(email, [])
    return {"history": list(reversed(history[-50:]))}

@app.get("/api/saved")
async def get_saved_judgments(email: str = Depends(get_current_user)):
    saved = saved_judgments_db.get(email, [])
    return {"saved": list(reversed(saved))}

@app.post("/api/save-judgment")
async def save_judgment(request: SaveJudgmentRequest, email: str = Depends(get_current_user)):
    saved_item = {
        "id": str(uuid.uuid4()),
        "file_name": request.file_name,
        "text": request.text,
        "score": request.score,
        "saved_at": datetime.now().isoformat(),
    }
    saved_judgments_db[email].append(saved_item)
    save_db(SAVED_JUDGMENTS_DB_FILE, saved_judgments_db)
    logger.info("Judgment saved | user=%s | file=%s", email, request.file_name)
    return {"message": "Judgment saved successfully", "id": saved_item["id"]}

# ============================================================
# API ENDPOINTS — FILE UPLOAD & INDEXING
# ============================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), email: str = Depends(get_current_user)):
    original_name = file.filename or ""
    file_name = os.path.basename(original_name.replace("\\", "/")).strip()
    logger.info("File upload | user=%s | filename=%s", email, file_name or "<missing>")

    if not file_name:
        raise HTTPException(status_code=400, detail="A file name is required.")

    extension = os.path.splitext(file_name)[1].lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")

    try:
        content = await file.read()
    except Exception:
        logger.warning("Could not read upload for user=%s:\n%s", email, traceback.format_exc())
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. The maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        if extension == ".pdf":
            text = extract_text_from_pdf(content)
        elif extension == ".docx":
            text = extract_text_from_docx(content)
        else:
            text = content.decode("utf-8")
    except HTTPException:
        raise
    except Exception:
        logger.warning("Text extraction failed for user=%s | filename=%s:\n%s", email, file_name, traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this file. Make sure it is a valid, non-encrypted document.",
        )

    text = (text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="No readable text could be extracted from this file.",
        )

    try:
        indexed_chunks = index_uploaded_document(text, file_name, email)
        if indexed_chunks <= 0:
            raise HTTPException(
                status_code=400,
                detail="No readable text could be indexed from this file.",
            )
        with _upload_lock:
            save_db(UPLOADED_DOCS_DB_FILE, uploaded_documents_db)
    except HTTPException:
        raise
    except Exception:
        logger.error("Error in /api/upload for user=%s:\n%s", email, traceback.format_exc())
        raise HTTPException(status_code=500, detail="File upload/indexing failed. Check server logs.")

    logger.info(
        "File indexed | user=%s | filename=%s | text_length=%d | chunks=%d",
        email, file_name, len(text), indexed_chunks,
    )
    return {
        "message": "File uploaded and indexed successfully",
        "filename": file_name,
        "text_length": len(text),
        "chunks": indexed_chunks,
    }

# ============================================================
# FRONTEND — HTML/CSS/JS (unchanged)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Search and analyze Peshawar High Court criminal judgments with source-grounded AI assistance.">
        <meta name="theme-color" content="#101a16">
        <title>PHC Legal Research · Judgment Intelligence</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                --font-serif: Georgia, 'Times New Roman', serif;
                --bg: #f6f7f5;
                --surface: #ffffff;
                --surface-2: #f1f3f1;
                --surface-3: #e9ece9;
                --ink: #18211d;
                --ink-2: #46524c;
                --muted: #75817a;
                --line: #e1e5e1;
                --line-strong: #cbd3cd;
                --sidebar: #101a16;
                --sidebar-2: #16231e;
                --sidebar-line: rgba(255, 255, 255, 0.08);
                --sidebar-text: #e6ede9;
                --sidebar-muted: #8fa198;
                --accent: #17715f;
                --accent-strong: #0d5849;
                --accent-soft: #e5f2ed;
                --accent-line: #c7e3d9;
                --gold: #a9782f;
                --gold-soft: #fbf7ee;
                --danger: #b42318;
                --danger-soft: #fef3f2;
                --success: #067647;
                --shadow-xs: 0 1px 2px rgba(18, 28, 23, 0.04);
                --shadow-sm: 0 2px 8px rgba(18, 28, 23, 0.05);
                --shadow-md: 0 14px 40px rgba(18, 28, 23, 0.09);
                --radius-sm: 8px;
                --radius-md: 12px;
                --radius-lg: 18px;
                --radius-xl: 26px;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            html { height: 100%; }

            body {
                height: 100%;
                overflow: hidden;
                font-family: var(--font-sans);
                background: var(--bg);
                color: var(--ink);
                font-size: 15px;
                line-height: 1.5;
                -webkit-font-smoothing: antialiased;
                text-rendering: optimizeLegibility;
            }

            button, input, textarea { font: inherit; color: inherit; }
            button { cursor: pointer; }
            button:disabled { cursor: not-allowed; }
            a { color: inherit; }
            ul { list-style: none; }
            .hidden { display: none !important; }

            ::selection { background: #c7e3d9; color: #0d3a30; }

            :focus-visible {
                outline: 2px solid var(--accent);
                outline-offset: 2px;
            }

            * {
                scrollbar-width: thin;
                scrollbar-color: #cbd3cd transparent;
            }
            *::-webkit-scrollbar { width: 8px; height: 8px; }
            *::-webkit-scrollbar-track { background: transparent; }
            *::-webkit-scrollbar-thumb {
                background: #cbd3cd;
                border-radius: 999px;
                border: 2px solid transparent;
                background-clip: content-box;
            }

            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }

            /* ============================================================
               AUTHENTICATION
               ============================================================ */

            .auth-screen {
                min-height: 100dvh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 32px;
                overflow-y: auto;
                background:
                    radial-gradient(circle at 12% 12%, rgba(23, 113, 95, 0.10), transparent 34%),
                    radial-gradient(circle at 88% 82%, rgba(169, 120, 47, 0.08), transparent 30%),
                    #eef1ee;
            }

            .auth-shell {
                width: min(1040px, 100%);
                min-height: 640px;
                display: grid;
                grid-template-columns: 1.05fr 0.95fr;
                background: var(--surface);
                border: 1px solid rgba(203, 211, 205, 0.9);
                border-radius: var(--radius-xl);
                overflow: hidden;
                box-shadow: 0 30px 80px rgba(16, 26, 22, 0.14);
            }

            .auth-brand-panel {
                position: relative;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 42px;
                padding: 48px;
                color: #f4f8f6;
                background:
                    radial-gradient(circle at 82% 12%, rgba(70, 160, 135, 0.32), transparent 36%),
                    radial-gradient(circle at 15% 88%, rgba(169, 120, 47, 0.16), transparent 34%),
                    linear-gradient(150deg, #0f201a 0%, #143a2f 68%, #10251d 100%);
                overflow: hidden;
            }

            .auth-brand-panel::after {
                content: '';
                position: absolute;
                width: 340px;
                height: 340px;
                right: -170px;
                bottom: -180px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 50%;
                box-shadow: 0 0 0 48px rgba(255, 255, 255, 0.035), 0 0 0 96px rgba(255, 255, 255, 0.025);
                pointer-events: none;
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 12px;
                position: relative;
                z-index: 1;
            }

            .brand-mark {
                width: 40px;
                height: 40px;
                display: grid;
                place-items: center;
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(255, 255, 255, 0.16);
                color: #d8eee6;
                font-size: 1.05rem;
            }

            .brand-name {
                display: flex;
                flex-direction: column;
                line-height: 1.12;
            }

            .brand-name strong {
                font-size: 1rem;
                font-weight: 650;
                letter-spacing: -0.01em;
            }

            .brand-name span {
                font-size: 0.72rem;
                color: rgba(230, 237, 233, 0.64);
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .auth-brand-copy {
                position: relative;
                z-index: 1;
                max-width: 440px;
            }

            .auth-brand-copy .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 20px;
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #a9c7bc;
            }

            .auth-brand-copy .eyebrow::before {
                content: '';
                width: 22px;
                height: 1px;
                background: currentColor;
                opacity: 0.7;
            }

            .auth-brand-copy h1 {
                font-family: var(--font-serif);
                font-size: clamp(2rem, 3.4vw, 2.65rem);
                font-weight: 500;
                line-height: 1.15;
                letter-spacing: -0.025em;
                color: #ffffff;
            }

            .auth-brand-copy p {
                margin-top: 18px;
                max-width: 380px;
                color: rgba(230, 237, 233, 0.74);
                font-size: 0.94rem;
                line-height: 1.7;
            }

            .auth-features {
                display: grid;
                gap: 14px;
                position: relative;
                z-index: 1;
            }

            .auth-feature {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                color: rgba(230, 237, 233, 0.82);
                font-size: 0.86rem;
            }

            .auth-feature i {
                width: 28px;
                height: 28px;
                display: grid;
                place-items: center;
                flex: 0 0 auto;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.10);
                color: #b7d8cc;
                font-size: 0.75rem;
            }

            .auth-feature strong {
                display: block;
                color: #f1f7f4;
                font-weight: 600;
                margin-bottom: 1px;
            }

            .auth-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                padding: 56px 48px;
                background: var(--surface);
            }

            #loginForm, #registerForm {
                width: 100%;
                max-width: 380px;
                margin: 0 auto;
            }

            .auth-header { margin-bottom: 30px; }

            .auth-header h2 {
                font-size: 1.65rem;
                font-weight: 650;
                letter-spacing: -0.03em;
                color: var(--ink);
            }

            .auth-header p {
                margin-top: 8px;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .form-group { margin-bottom: 18px; }

            .form-group label {
                display: block;
                margin-bottom: 7px;
                color: var(--ink-2);
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.01em;
            }

            .form-group input {
                width: 100%;
                height: 46px;
                padding: 0 14px;
                background: #fbfcfb;
                border: 1px solid var(--line-strong);
                border-radius: 10px;
                font-size: 0.92rem;
                transition: border-color 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
            }

            .form-group input:hover { border-color: #b5c0b9; }
            .form-group input:focus {
                outline: none;
                background: var(--surface);
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(23, 113, 95, 0.12);
            }
            .form-group input::placeholder { color: #a0aaa4; }

            .btn-auth {
                width: 100%;
                height: 46px;
                margin-top: 6px;
                border: 1px solid transparent;
                border-radius: 10px;
                background: var(--accent);
                color: #fff;
                font-size: 0.9rem;
                font-weight: 650;
                box-shadow: 0 2px 4px rgba(13, 88, 73, 0.15);
                transition: background 0.16s ease, transform 0.16s ease, box-shadow 0.16s ease;
            }

            .btn-auth:hover {
                background: var(--accent-strong);
                box-shadow: 0 4px 12px rgba(13, 88, 73, 0.2);
            }

            .btn-auth:active { transform: translateY(1px); }

            .auth-switch {
                margin-top: 20px;
                text-align: center;
                color: var(--muted);
                font-size: 0.85rem;
            }

            .auth-switch a {
                color: var(--accent);
                font-weight: 600;
                text-decoration: none;
                cursor: pointer;
            }

            .auth-switch a:hover { text-decoration: underline; text-underline-offset: 3px; }

            .auth-note {
                margin-top: 28px;
                padding-top: 20px;
                border-top: 1px solid var(--line);
                color: var(--muted);
                font-size: 0.73rem;
                line-height: 1.6;
                text-align: center;
            }

            /* ============================================================
               APPLICATION SHELL
               ============================================================ */

            .app-container {
                display: none;
                height: 100dvh;
                overflow: hidden;
                background: var(--bg);
            }

            .app-layout { display: flex; height: 100%; }

            /* ============================================================
               SIDEBAR
               ============================================================ */

            .sidebar {
                width: 300px;
                flex: 0 0 300px;
                display: flex;
                flex-direction: column;
                background: var(--sidebar);
                color: var(--sidebar-text);
                border-right: 1px solid rgba(255, 255, 255, 0.04);
            }

            .sidebar-header {
                padding: 22px 18px 16px;
                border-bottom: 1px solid var(--sidebar-line);
            }

            .sidebar-header .brand-lockup { margin-bottom: 18px; padding: 0 4px; }

            .sidebar-header .brand-mark {
                width: 34px;
                height: 34px;
                border-radius: 9px;
                background: rgba(28, 128, 106, 0.18);
                border-color: rgba(86, 186, 159, 0.22);
                color: #9dd7c5;
                font-size: 0.9rem;
            }

            .sidebar-header .brand-name strong { color: #f3f8f5; font-size: 0.92rem; }
            .sidebar-header .brand-name span { color: #7f948b; font-size: 0.65rem; }

            .new-chat-btn {
                width: 100%;
                height: 42px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 9px;
                background: var(--accent);
                border: 1px solid rgba(255, 255, 255, 0.10);
                color: #ffffff;
                border-radius: 10px;
                font-size: 0.86rem;
                font-weight: 600;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.18);
                transition: background 0.16s ease, transform 0.16s ease;
            }

            .new-chat-btn:hover { background: #1a8270; }
            .new-chat-btn:active { transform: translateY(1px); }
            .new-chat-btn i { font-size: 0.8rem; }

            .sidebar-content {
                flex: 1;
                overflow-y: auto;
                padding: 16px 12px 24px;
            }

            .sidebar-section { margin-bottom: 22px; }

            .sidebar-section h3 {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 9px;
                margin-bottom: 8px;
                color: var(--sidebar-muted);
                font-size: 0.66rem;
                font-weight: 650;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }

            .sidebar-empty {
                padding: 10px 9px;
                color: #7f9189;
                font-size: 0.78rem;
                line-height: 1.5;
            }

            .history-item {
                position: relative;
                padding: 10px 11px;
                margin-bottom: 3px;
                border: 1px solid transparent;
                border-radius: 9px;
                cursor: pointer;
                transition: background 0.14s ease, border-color 0.14s ease;
            }

            .history-item:hover { background: rgba(255, 255, 255, 0.055); }

            .history-item.active {
                background: rgba(255, 255, 255, 0.085);
                border-color: rgba(255, 255, 255, 0.07);
            }

            .history-item.active::before {
                content: '';
                position: absolute;
                left: -1px;
                top: 10px;
                bottom: 10px;
                width: 2px;
                border-radius: 2px;
                background: #54b69d;
            }

            .history-item-query {
                color: #e8efeb;
                font-size: 0.82rem;
                line-height: 1.4;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .history-item-time {
                margin-top: 3px;
                color: #84978e;
                font-size: 0.69rem;
                line-height: 1.4;
            }

            .context-menu {
                position: fixed;
                z-index: 3000;
                min-width: 168px;
                padding: 5px;
                border: 1px solid var(--line);
                border-radius: 10px;
                background: var(--surface);
                box-shadow: 0 14px 36px rgba(16, 26, 22, 0.22);
                display: none;
            }

            .context-menu.open { display: block; }

            .context-menu-item {
                width: 100%;
                display: flex;
                align-items: center;
                gap: 9px;
                padding: 9px 10px;
                border: 0;
                border-radius: 7px;
                background: transparent;
                color: var(--ink);
                font-family: inherit;
                font-size: 0.82rem;
                text-align: left;
                cursor: pointer;
            }

            .context-menu-item:hover { background: var(--surface-2); }

            .context-menu-item.danger { color: #a33131; }

            .context-menu-item.danger:hover { background: #f8e9e9; }

            .sidebar-footer {
                padding: 12px;
                border-top: 1px solid var(--sidebar-line);
                background: rgba(0, 0, 0, 0.10);
            }

            .user-profile {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 9px 10px;
                border: 1px solid transparent;
                border-radius: 10px;
                cursor: pointer;
                transition: background 0.14s ease, border-color 0.14s ease;
            }

            .user-profile:hover {
                background: rgba(255, 255, 255, 0.055);
                border-color: rgba(255, 255, 255, 0.06);
            }

            .user-avatar {
                width: 34px;
                height: 34px;
                flex: 0 0 auto;
                display: grid;
                place-items: center;
                border-radius: 50%;
                background: linear-gradient(145deg, #2c8f79, #185c4e);
                color: #ffffff;
                font-size: 0.82rem;
                font-weight: 650;
            }

            .user-info { flex: 1; min-width: 0; }

            .user-name {
                color: #eef4f1;
                font-size: 0.79rem;
                font-weight: 600;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .user-email {
                margin-top: 1px;
                color: #83968d;
                font-size: 0.67rem;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .user-profile > .fa-sign-out-alt {
                color: #7f9289;
                font-size: 0.78rem;
            }

            .sidebar-overlay { display: none; }

            /* ============================================================
               MAIN CONTENT
               ============================================================ */

            .main-content {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                background: var(--bg);
            }

            .chat-header {
                height: 72px;
                flex: 0 0 72px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                padding: 0 clamp(20px, 3vw, 34px);
                background: rgba(255, 255, 255, 0.88);
                border-bottom: 1px solid var(--line);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                z-index: 5;
            }

            .header-left {
                display: flex;
                align-items: center;
                gap: 10px;
                min-width: 0;
            }

            .icon-btn {
                display: none;
                width: 36px;
                height: 36px;
                align-items: center;
                justify-content: center;
                border: 1px solid var(--line);
                border-radius: 9px;
                background: var(--surface);
                color: var(--ink-2);
                transition: background 0.15s ease, border-color 0.15s ease;
            }

            .icon-btn:hover { background: var(--surface-2); border-color: var(--line-strong); }

            .chat-title {
                font-size: 0.98rem;
                font-weight: 650;
                letter-spacing: -0.015em;
                color: var(--ink);
            }

            .chat-subtitle {
                margin-top: 2px;
                color: var(--muted);
                font-size: 0.74rem;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                height: 28px;
                padding: 0 10px;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: #fbfcfb;
                color: var(--ink-2);
                font-size: 0.69rem;
                font-weight: 550;
                white-space: nowrap;
            }

            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #25a97e;
                box-shadow: 0 0 0 3px rgba(37, 169, 126, 0.13);
            }

            /* ============================================================
               CONVERSATION
               ============================================================ */

            .chat-messages {
                flex: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 34px clamp(20px, 4vw, 48px) 28px;
                scroll-behavior: smooth;
            }

            .welcome-screen {
                width: min(880px, 100%);
                padding: 24px 0 40px;
                text-align: center;
            }

            .welcome-icon {
                width: 56px;
                height: 56px;
                margin: 0 auto 18px;
                display: grid;
                place-items: center;
                border-radius: 16px;
                background: var(--accent-soft);
                border: 1px solid var(--accent-line);
                color: var(--accent);
                font-size: 1.25rem;
            }

            .welcome-eyebrow {
                display: block;
                margin-bottom: 12px;
                color: var(--accent);
                font-size: 0.69rem;
                font-weight: 650;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .welcome-screen h2 {
                font-family: var(--font-serif);
                color: var(--ink);
                font-size: clamp(1.75rem, 3.2vw, 2.35rem);
                font-weight: 500;
                line-height: 1.18;
                letter-spacing: -0.025em;
            }

            .welcome-screen > p {
                max-width: 620px;
                margin: 14px auto 30px;
                color: var(--ink-2);
                font-size: 0.92rem;
                line-height: 1.65;
            }

            .quick-prompts {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                width: 100%;
                max-width: 700px;
                margin: 0 auto;
                text-align: left;
            }

            .prompt-card {
                display: flex;
                align-items: flex-start;
                gap: 11px;
                min-height: 72px;
                padding: 13px 14px;
                border: 1px solid var(--line);
                border-radius: var(--radius-md);
                background: var(--surface);
                text-align: left;
                box-shadow: var(--shadow-xs);
                transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
            }

            .prompt-card:hover {
                border-color: var(--accent-line);
                box-shadow: var(--shadow-sm);
                transform: translateY(-1px);
            }

            .prompt-icon {
                width: 30px;
                height: 30px;
                flex: 0 0 auto;
                display: grid;
                place-items: center;
                border-radius: 8px;
                background: var(--surface-2);
                color: var(--accent);
                font-size: 0.72rem;
            }

            .prompt-copy { display: block; min-width: 0; }

            .prompt-title {
                display: block;
                color: var(--ink);
                font-size: 0.79rem;
                font-weight: 600;
                line-height: 1.35;
            }

            .prompt-text {
                display: block;
                margin-top: 2px;
                color: var(--muted);
                font-size: 0.71rem;
                line-height: 1.4;
            }

            .message {
                width: min(880px, 100%);
                margin-bottom: 22px;
                animation: messageIn 0.2s ease both;
            }

            @keyframes messageIn {
                from { opacity: 0; transform: translateY(5px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            .message-user {
                width: auto;
                max-width: min(720px, 82%);
                align-self: flex-end;
            }

            .message-assistant { align-self: center; }

            .message-content {
                color: var(--ink);
                font-size: 0.92rem;
                line-height: 1.7;
            }

            .message-user .message-content {
                padding: 12px 17px;
                background: var(--accent);
                color: #ffffff;
                border: 1px solid rgba(13, 88, 73, 0.14);
                border-radius: 17px 17px 5px 17px;
                box-shadow: var(--shadow-xs);
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }

            .message-assistant .message-content {
                padding: 16px 19px;
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 5px 17px 17px 17px;
                box-shadow: var(--shadow-xs);
                overflow-wrap: anywhere;
            }

            .message-with-card .message-content,
            .message-assistant .message-content:has(.judgment-card) {
                padding: 0;
                background: transparent;
                border: 0;
                box-shadow: none;
            }

            .message-assistant .message-content p + p { margin-top: 10px; }
            .message-assistant .message-content ul { margin: 8px 0 0; padding-left: 20px; list-style: disc; }
            .message-assistant .message-content li + li { margin-top: 4px; }
            .message-assistant .message-content strong { font-weight: 650; }

            /* ============================================================
               JUDGMENT CARDS
               ============================================================ */

            .judgment-card {
                margin-bottom: 10px;
                padding: 18px 20px;
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius-lg);
                box-shadow: var(--shadow-sm);
            }

            .judgment-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 14px;
            }

            .judgment-heading {
                display: flex;
                align-items: flex-start;
                gap: 11px;
                min-width: 0;
            }

            .judgment-doc-icon {
                width: 34px;
                height: 34px;
                flex: 0 0 auto;
                display: grid;
                place-items: center;
                border-radius: 9px;
                background: var(--accent-soft);
                border: 1px solid var(--accent-line);
                color: var(--accent);
                font-size: 0.8rem;
            }

            .judgment-kicker {
                color: var(--accent);
                font-size: 0.64rem;
                font-weight: 700;
                letter-spacing: 0.11em;
                text-transform: uppercase;
            }

            .judgment-file {
                margin-top: 3px;
                color: var(--ink);
                font-size: 0.86rem;
                font-weight: 600;
                line-height: 1.45;
                overflow-wrap: anywhere;
            }

            .judgment-meta {
                margin-top: 2px;
                color: var(--muted);
                font-size: 0.7rem;
            }

            .score-badge {
                flex: 0 0 auto;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                height: 27px;
                padding: 0 9px;
                border: 1px solid var(--accent-line);
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent-strong);
                font-size: 0.69rem;
                font-weight: 650;
                white-space: nowrap;
            }

            .score-badge i { font-size: 0.64rem; }

            .judgment-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 7px;
            }

            .action-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 7px;
                height: 33px;
                padding: 0 11px;
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                color: var(--ink-2);
                font-size: 0.75rem;
                font-weight: 550;
                transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease;
            }

            .action-btn:hover {
                background: var(--surface-2);
                border-color: var(--line-strong);
                color: var(--ink);
            }

            .action-btn:disabled {
                opacity: 0.55;
                cursor: wait;
            }

            .action-btn i { font-size: 0.7rem; color: var(--muted); }
            .action-btn:hover i { color: var(--accent); }

            .action-btn.active {
                background: var(--accent);
                border-color: var(--accent);
                color: #ffffff;
            }

            .action-btn.active i { color: rgba(255, 255, 255, 0.8); }

            .judgment-text {
                margin-top: 14px;
                padding: 16px;
                max-height: 420px;
                overflow-y: auto;
                background: #fbfcfb;
                border: 1px solid var(--line);
                border-radius: 10px;
                color: #303b35;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                font-size: 0.76rem;
                line-height: 1.7;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }

            .summary-box {
                margin-top: 14px;
                padding: 18px 20px;
                background: var(--gold-soft);
                border: 1px solid #ece2ce;
                border-left: 3px solid var(--gold);
                border-radius: 10px;
            }

            .summary-box-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 12px;
            }

            .summary-box-title {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: #6d4c17;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .summary-box-title i { color: var(--gold); }

            .summary-content {
                color: #3c3427;
                font-size: 0.87rem;
                line-height: 1.7;
                overflow-wrap: anywhere;
            }

            .summary-content h4 {
                margin: 14px 0 5px;
                color: #4d3a19;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .summary-content h4:first-child { margin-top: 0; }

            .summary-content p { margin: 0 0 8px; }
            .summary-content p:last-child { margin-bottom: 0; }

            .summary-content ul {
                margin: 0 0 9px;
                padding-left: 19px;
                list-style: disc;
            }

            .summary-content li { margin-bottom: 3px; }
            .summary-content strong { color: #443418; font-weight: 650; }

            .summary-placeholder {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 14px;
                padding: 16px 18px;
                background: var(--gold-soft);
                border: 1px solid #ece2ce;
                border-left: 3px solid var(--gold);
                border-radius: 10px;
                color: #7a6540;
                font-size: 0.82rem;
            }

            /* ============================================================
               COMPOSER
               ============================================================ */

            .chat-input-container {
                flex: 0 0 auto;
                padding: 14px clamp(20px, 4vw, 48px) 16px;
                background: linear-gradient(to top, var(--bg) 70%, rgba(246, 247, 245, 0));
            }

            .chat-input-wrapper {
                width: min(880px, 100%);
                margin: 0 auto;
                display: flex;
                align-items: flex-end;
                gap: 5px;
                padding: 7px;
                background: var(--surface);
                border: 1px solid var(--line-strong);
                border-radius: 15px;
                box-shadow: 0 8px 28px rgba(18, 28, 23, 0.07);
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }

            .chat-input-wrapper:focus-within {
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(23, 113, 95, 0.10), 0 8px 28px rgba(18, 28, 23, 0.07);
            }

            .input-upload-btn {
                width: 38px;
                height: 38px;
                flex: 0 0 auto;
                display: grid;
                place-items: center;
                border: 0;
                border-radius: 9px;
                background: transparent;
                color: var(--muted);
                font-size: 0.9rem;
                transition: background 0.14s ease, color 0.14s ease;
            }

            .input-upload-btn:hover {
                background: var(--surface-2);
                color: var(--accent);
            }

            .chat-input {
                flex: 1;
                min-width: 0;
                min-height: 38px;
                max-height: 168px;
                padding: 8px 5px;
                border: 0;
                outline: 0;
                resize: none;
                background: transparent;
                color: var(--ink);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .chat-input::placeholder { color: #94a09a; }

            .input-send-btn {
                height: 38px;
                padding: 0 14px;
                flex: 0 0 auto;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 7px;
                border: 1px solid transparent;
                border-radius: 9px;
                background: var(--accent);
                color: #ffffff;
                font-size: 0.8rem;
                font-weight: 600;
                transition: background 0.14s ease, opacity 0.14s ease, transform 0.14s ease;
            }

            .input-send-btn:hover { background: var(--accent-strong); }
            .input-send-btn:active { transform: translateY(1px); }
            .input-send-btn:disabled { opacity: 0.55; }

            .input-send-btn i { font-size: 0.72rem; }

            .composer-meta {
                width: min(880px, 100%);
                margin: 8px auto 0;
                display: flex;
                justify-content: space-between;
                gap: 16px;
                color: var(--muted);
                font-size: 0.68rem;
                line-height: 1.5;
            }

            /* ============================================================
               LOADING / NOTIFICATIONS
               ============================================================ */

            .loading-message {
                display: flex;
                align-items: center;
                gap: 11px;
                color: var(--ink-2);
                font-size: 0.84rem;
            }

            .loading-dots {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                flex: 0 0 auto;
            }

            .loading-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: var(--accent);
                animation: dotPulse 1.2s infinite ease-in-out;
            }

            .loading-dot:nth-child(2) { animation-delay: 0.14s; }
            .loading-dot:nth-child(3) { animation-delay: 0.28s; }

            @keyframes dotPulse {
                0%, 70%, 100% { opacity: 0.28; transform: translateY(0); }
                35% { opacity: 1; transform: translateY(-2px); }
            }

            .loading-elapsed {
                margin-left: auto;
                color: var(--muted);
                font-size: 0.7rem;
                font-variant-numeric: tabular-nums;
            }

            .notification {
                position: fixed;
                top: 18px;
                right: 18px;
                z-index: 1000;
                display: flex;
                align-items: flex-start;
                gap: 10px;
                max-width: min(420px, calc(100vw - 36px));
                padding: 12px 15px;
                background: #121b17;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                color: #f1f6f3;
                box-shadow: var(--shadow-md);
                font-size: 0.82rem;
                line-height: 1.45;
                animation: toastIn 0.18s ease both;
            }

            .notification i { margin-top: 2px; }
            .notification.success { border-left: 3px solid #34b58d; }
            .notification.success i { color: #6dd6b5; }
            .notification.error { border-left: 3px solid #f97066; }
            .notification.error i { color: #fda29b; }

            @keyframes toastIn {
                from { opacity: 0; transform: translateY(-6px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            /* ============================================================
               RESPONSIVE
               ============================================================ */

            @media (max-width: 960px) {
                .auth-shell {
                    grid-template-columns: 1fr;
                    max-width: 480px;
                    min-height: 0;
                }

                .auth-brand-panel {
                    padding: 30px 32px;
                    gap: 18px;
                }

                .auth-brand-copy h1 { font-size: 1.65rem; }
                .auth-brand-copy p { margin-top: 10px; font-size: 0.86rem; }
                .auth-brand-copy .eyebrow { margin-bottom: 12px; }
                .auth-features { display: none; }
                .auth-container { padding: 34px 32px 38px; }

                .sidebar {
                    position: fixed;
                    inset: 0 auto 0 0;
                    z-index: 60;
                    width: 290px;
                    flex-basis: 290px;
                    transform: translateX(-102%);
                    transition: transform 0.22s ease;
                    box-shadow: 20px 0 50px rgba(8, 16, 12, 0.22);
                }

                .sidebar.open { transform: translateX(0); }

                .sidebar-overlay {
                    display: block;
                    position: fixed;
                    inset: 0;
                    z-index: 55;
                    background: rgba(8, 16, 12, 0.46);
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.22s ease;
                }

                .sidebar-overlay.visible {
                    opacity: 1;
                    pointer-events: auto;
                }

                .icon-btn { display: inline-flex; }
                .chat-header { padding: 0 18px; }
                .chat-messages { padding: 28px 18px 20px; }
                .chat-input-container { padding: 12px 18px 14px; }
            }

            @media (max-width: 620px) {
                .auth-screen { padding: 14px; align-items: flex-start; }
                .auth-shell { border-radius: 20px; }
                .auth-brand-panel { padding: 24px; }
                .auth-container { padding: 28px 24px 30px; }
                .auth-brand-copy h1 { font-size: 1.45rem; }
                .auth-brand-copy p { display: none; }

                .chat-header { height: 64px; flex-basis: 64px; }
                .chat-subtitle { display: none; }
                .status-pill span:last-child { display: none; }
                .status-pill { width: 28px; padding: 0; justify-content: center; }

                .chat-messages { padding: 22px 14px 16px; }
                .welcome-screen { padding-top: 12px; }
                .welcome-screen > p { font-size: 0.86rem; margin-bottom: 22px; }
                .quick-prompts { grid-template-columns: 1fr; }
                .prompt-card { min-height: 0; }

                .message-user { max-width: 92%; }
                .message-content { font-size: 0.9rem; }
                .message-assistant .message-content { padding: 14px 15px; }
                .judgment-card { padding: 15px 15px; }
                .judgment-header { flex-direction: column; gap: 10px; }
                .score-badge { align-self: flex-start; }
                .action-btn { flex: 1 1 auto; }
                .summary-box { padding: 15px; }

                .chat-input-container { padding: 10px 12px 12px; }
                .input-send-btn { width: 38px; padding: 0; }
                .input-send-btn .send-label { display: none; }
                .composer-meta { display: none; }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
                }
            }
        </style>
    </head>
    <body>
        <div id="authScreen" class="auth-screen">
            <div class="auth-shell">
                <section class="auth-brand-panel">
                    <div class="brand-lockup">
                        <div class="brand-mark"><i class="fas fa-scale-balanced"></i></div>
                        <div class="brand-name">
                            <strong>PHC Legal Research</strong>
                            <span>Judgment Intelligence</span>
                        </div>
                    </div>
                    <div class="auth-brand-copy">
                        <span class="eyebrow">Grounded legal research</span>
                        <h1>Find the right precedent, with the source in view.</h1>
                        <p>Search thousands of Peshawar High Court criminal judgments, review the strongest passages, and develop case-specific follow-up questions.</p>
                    </div>
                    <div class="auth-features">
                        <div class="auth-feature">
                            <i class="fas fa-database"></i>
                            <div><strong>2,000+ judgments indexed</strong>Full-text criminal archive with hybrid retrieval.</div>
                        </div>
                        <div class="auth-feature">
                            <i class="fas fa-list-check"></i>
                            <div><strong>Structured case summaries</strong>Issue, facts, holding, and reasoning at a glance.</div>
                        </div>
                        <div class="auth-feature">
                            <i class="fas fa-comments"></i>
                            <div><strong>Source-grounded discussion</strong>Ask follow-up questions without leaving the judgment.</div>
                        </div>
                    </div>
                </section>

                <div class="auth-container">
                    <div id="loginForm">
                        <div class="auth-header">
                            <h2>Welcome back</h2>
                            <p>Sign in to continue your legal research.</p>
                        </div>
                        <div class="form-group">
                            <label for="loginEmail">Email</label>
                            <input type="email" id="loginEmail" placeholder="your@email.com" autocomplete="email" onkeydown="if(event.key==='Enter') handleLogin()">
                        </div>
                        <div class="form-group">
                            <label for="loginPassword">Password</label>
                            <input type="password" id="loginPassword" placeholder="••••••••" autocomplete="current-password" onkeydown="if(event.key==='Enter') handleLogin()">
                        </div>
                        <button class="btn-auth" type="button" onclick="handleLogin()">Sign in</button>
                        <div class="auth-switch">
                            New here? <a role="button" tabindex="0" onclick="showRegister()" onkeydown="if(event.key==='Enter') showRegister()">Create an account</a>
                        </div>
                    </div>

                    <div id="registerForm" class="hidden">
                        <div class="auth-header">
                            <h2>Create your account</h2>
                            <p>Start researching the criminal judgment archive.</p>
                        </div>
                        <div class="form-group">
                            <label for="registerName">Full name</label>
                            <input type="text" id="registerName" placeholder="John Doe" autocomplete="name">
                        </div>
                        <div class="form-group">
                            <label for="registerEmail">Email</label>
                            <input type="email" id="registerEmail" placeholder="your@email.com" autocomplete="email">
                        </div>
                        <div class="form-group">
                            <label for="registerPassword">Password</label>
                            <input type="password" id="registerPassword" placeholder="••••••••" autocomplete="new-password" onkeydown="if(event.key==='Enter') handleRegister()">
                        </div>
                        <button class="btn-auth" type="button" onclick="handleRegister()">Create account</button>
                        <div class="auth-switch">
                            Already have an account? <a role="button" tabindex="0" onclick="showLogin()" onkeydown="if(event.key==='Enter') showLogin()">Sign in</a>
                        </div>
                    </div>

                    <p class="auth-note">Research assistance only. Always verify findings against the original judgment before relying on them.</p>
                </div>
            </div>
        </div>

        <div id="appContainer" class="app-container">
            <div class="app-layout">
                <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()" aria-hidden="true"></div>
                <aside class="sidebar" id="sidebar">
                    <div class="sidebar-header">
                        <div class="brand-lockup">
                            <div class="brand-mark"><i class="fas fa-scale-balanced"></i></div>
                            <div class="brand-name">
                                <strong>PHC Legal Research</strong>
                                <span>Judgment Intelligence</span>
                            </div>
                        </div>
                        <button class="new-chat-btn" type="button" onclick="startNewChat()">
                            <i class="fas fa-plus"></i>
                            <span>New research chat</span>
                        </button>
                    </div>

                    <div class="sidebar-content">
                        <div class="sidebar-section">
                            <h3>Recent chats</h3>
                            <div id="historyList"></div>
                        </div>
                        <div class="sidebar-section">
                            <h3>Saved judgments</h3>
                            <div id="savedList"></div>
                        </div>
                    </div>

                    <div class="sidebar-footer">
                        <div class="user-profile" role="button" tabindex="0" title="Sign out" onclick="handleLogout()" onkeydown="if(event.key==='Enter') handleLogout()">
                            <div class="user-avatar" id="userAvatar">U</div>
                            <div class="user-info">
                                <div class="user-name" id="userName">User</div>
                                <div class="user-email" id="userEmail">user@email.com</div>
                            </div>
                            <i class="fas fa-sign-out-alt"></i>
                        </div>
                    </div>
                </aside>

                <main class="main-content">
                    <header class="chat-header">
                        <div class="header-left">
                            <button class="icon-btn" id="sidebarToggle" type="button" aria-label="Toggle sidebar" aria-expanded="false" onclick="toggleSidebar()">
                                <i class="fas fa-bars"></i>
                            </button>
                            <div>
                                <div class="chat-title">Judgment library</div>
                                <div class="chat-subtitle">Peshawar High Court criminal judgments</div>
                            </div>
                        </div>
                        <span class="status-pill">
                            <span class="status-dot"></span>
                            <span>Index online</span>
                        </span>
                    </header>

                    <div class="chat-messages" id="chatMessages">
                        <div class="welcome-screen" id="welcomeScreen">
                            <div class="welcome-icon"><i class="fas fa-scale-balanced"></i></div>
                            <span class="welcome-eyebrow">PHC criminal judgment archive</span>
                            <h2>What legal question can I help you research?</h2>
                            <p>Search the indexed Peshawar High Court criminal judgments, review the strongest passages, and ask follow-up questions grounded in the source text.</p>
                            <div class="quick-prompts">
                                <button class="prompt-card" type="button" onclick="usePrompt('Murder over a property dispute')">
                                    <span class="prompt-icon"><i class="fas fa-gavel"></i></span>
                                    <span class="prompt-copy">
                                        <span class="prompt-title">Property-linked murder</span>
                                        <span class="prompt-text">Find judgments involving murder over land or property.</span>
                                    </span>
                                </button>
                                <button class="prompt-card" type="button" onclick="usePrompt('Bail in narcotics cases')">
                                    <span class="prompt-icon"><i class="fas fa-unlock"></i></span>
                                    <span class="prompt-copy">
                                        <span class="prompt-title">Bail and narcotics</span>
                                        <span class="prompt-text">Review custody and bail reasoning in narcotics matters.</span>
                                    </span>
                                </button>
                                <button class="prompt-card" type="button" onclick="usePrompt('Appreciation of eyewitness evidence')">
                                    <span class="prompt-icon"><i class="fas fa-user-shield"></i></span>
                                    <span class="prompt-copy">
                                        <span class="prompt-title">Eyewitness evidence</span>
                                        <span class="prompt-text">Compare how courts assess credibility and contradictions.</span>
                                    </span>
                                </button>
                                <button class="prompt-card" type="button" onclick="usePrompt('Sentencing principles for murder')">
                                    <span class="prompt-icon"><i class="fas fa-scale-unbalanced"></i></span>
                                    <span class="prompt-copy">
                                        <span class="prompt-title">Sentencing principles</span>
                                        <span class="prompt-text">Explore aggravating and mitigating factors in murder sentencing.</span>
                                    </span>
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="chat-input-container">
                        <div class="chat-input-wrapper">
                            <button class="input-upload-btn" type="button" onclick="triggerFileUpload()" aria-label="Upload a judgment">
                                <i class="fas fa-paperclip"></i>
                            </button>
                            <input type="file" id="fileUpload" class="hidden" accept=".pdf,.txt,.docx" onchange="handleFileUpload(event)">
                            <textarea
                                id="chatInput"
                                class="chat-input"
                                placeholder="Enter your legal query..."
                                rows="1"
                                aria-label="Legal query"
                                onkeypress="handleKeyPress(event)"
                                oninput="autoGrowInput()"
                            ></textarea>
                            <button class="input-send-btn" id="sendButton" type="button" onclick="handleSend()">
                                <span class="send-label">Send</span>
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        <div class="composer-meta">
                            <span>Enter to send · Shift + Enter for a new line</span>
                            <span>Verify important findings against the source.</span>
                        </div>
                    </div>
                </main>
            </div>
        </div>

        <div id="chatContextMenu" class="context-menu" role="menu" aria-label="Chat actions">
            <button class="context-menu-item danger" type="button" role="menuitem" onclick="deleteChatFromMenu()">
                <i class="fas fa-trash"></i>
                <span>Delete</span>
            </button>
        </div>
        
        <script>
            // ── Global State ──────────────────────────────────────────
            let sessionId       = null;
            let currentUser     = null;
            let judgmentStore   = {};         // cardId → { fullText, fileName, score }
            let activeChatCardId = null;      // which judgment is in chat mode
            let currentTopK     = 3;
            let chatId          = null;       // active persisted chat
            let chats           = [];         // sidebar chat summaries
            let loadingTimer    = null;
            let contextChatId   = null;       // chat targeted by the context menu

            function toggleSidebar() {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebarOverlay');
                if (!sidebar || !overlay) return;

                const open = !sidebar.classList.contains('open');
                sidebar.classList.toggle('open', open);
                overlay.classList.toggle('visible', open);
                overlay.setAttribute('aria-hidden', String(!open));
                document.getElementById('sidebarToggle')?.setAttribute('aria-expanded', String(open));
            }

            function closeSidebar() {
                document.getElementById('sidebar')?.classList.remove('open');
                document.getElementById('sidebarOverlay')?.classList.remove('visible');
                document.getElementById('sidebarOverlay')?.setAttribute('aria-hidden', 'true');
                document.getElementById('sidebarToggle')?.setAttribute('aria-expanded', 'false');
            }

            function autoGrowInput() {
                const input = document.getElementById('chatInput');
                if (!input) return;
                input.style.height = 'auto';
                input.style.height = Math.min(input.scrollHeight, 168) + 'px';
            }

            function usePrompt(text) {
                const input = document.getElementById('chatInput');
                if (!input) return;
                input.value = text;
                autoGrowInput();
                input.focus();
            }

            // ── Auth Functions ────────────────────────────────────────
            function showRegister() {
                document.getElementById('loginForm').classList.add('hidden');
                document.getElementById('registerForm').classList.remove('hidden');
            }
            
            function showLogin() {
                document.getElementById('registerForm').classList.add('hidden');
                document.getElementById('loginForm').classList.remove('hidden');
            }
            
            async function handleRegister() {
                const name     = document.getElementById('registerName').value;
                const email    = document.getElementById('registerEmail').value;
                const password = document.getElementById('registerPassword').value;
                
                if (!name || !email || !password) {
                    showNotification('Please fill all fields', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/api/register', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password, full_name: name })
                    });
                    
                    if (response.ok) {
                        showNotification('Registration successful! Please login.', 'success');
                        showLogin();
                    } else {
                        const data = await response.json();
                        showNotification(data.detail || 'Registration failed', 'error');
                    }
                } catch (error) {
                    showNotification('Error: ' + error.message, 'error');
                }
            }
            
            async function handleLogin() {
                const email    = document.getElementById('loginEmail').value;
                const password = document.getElementById('loginPassword').value;
                
                if (!email || !password) {
                    showNotification('Please fill all fields', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        sessionId   = data.session_id;
                        currentUser = data;
                        
                        document.getElementById('authScreen').style.display  = 'none';
                        document.getElementById('appContainer').style.display = 'block';
                        document.getElementById('userName').textContent  = data.full_name;
                        document.getElementById('userEmail').textContent  = data.email;
                        document.getElementById('userAvatar').textContent = data.full_name.charAt(0).toUpperCase();
                        closeSidebar();
                        autoGrowInput();

                        const welcomeMessage = data.welcome_message || (data.is_new_user ? 'Welcome' : 'Welcome back');
                        showNotification(welcomeMessage + '!', 'success');

                        loadSavedJudgments();
                        await loadChats();
                        if (chats.length) {
                            await openChat(chats[0].id);
                        } else {
                            await startNewChat();
                        }
                    } else {
                        const data = await response.json();
                        showNotification(data.detail || 'Login failed', 'error');
                    }
                } catch (error) {
                    showNotification('Error: ' + error.message, 'error');
                }
            }
            
            async function handleLogout() {
                if (!confirm('Are you sure you want to logout?')) return;
                
                try {
                    await fetch('/api/logout', {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                } catch (error) {
                    console.error('Logout error:', error);
                }
                
                sessionId   = null;
                currentUser = null;
                closeSidebar();
                closeChatMenu();
                document.getElementById('appContainer').style.display = 'none';
                document.getElementById('authScreen').style.display   = 'flex';
                showNotification('Logged out successfully', 'success');
            }
            
            // ── Chat / Search Functions ───────────────────────────────
            async function startNewChat() {
                activeChatCardId = null;
                judgmentStore    = {};
                chatId           = null;

                document.getElementById('chatMessages').innerHTML = welcomeMarkup();
                document.getElementById('chatInput').value       = '';
                document.getElementById('chatInput').placeholder = 'Enter your legal query...';
                autoGrowInput();
                closeSidebar();
                await loadChats();
            }

            function welcomeMarkup() {
                return `
                    <div class="welcome-screen" id="welcomeScreen">
                        <div class="welcome-icon"><i class="fas fa-scale-balanced"></i></div>
                        <span class="welcome-eyebrow">PHC criminal judgment archive</span>
                        <h2>What legal question can I help you research?</h2>
                        <p>Search the indexed Peshawar High Court criminal judgments, review the strongest passages, and ask follow-up questions grounded in the source text.</p>
                        <div class="quick-prompts">
                            <button class="prompt-card" type="button" onclick="usePrompt('Murder over a property dispute')">
                                <span class="prompt-icon"><i class="fas fa-gavel"></i></span>
                                <span class="prompt-copy">
                                    <span class="prompt-title">Property-linked murder</span>
                                    <span class="prompt-text">Find judgments involving murder over land or property.</span>
                                </span>
                            </button>
                            <button class="prompt-card" type="button" onclick="usePrompt('Bail in narcotics cases')">
                                <span class="prompt-icon"><i class="fas fa-unlock"></i></span>
                                <span class="prompt-copy">
                                    <span class="prompt-title">Bail and narcotics</span>
                                    <span class="prompt-text">Review custody and bail reasoning in narcotics matters.</span>
                                </span>
                            </button>
                            <button class="prompt-card" type="button" onclick="usePrompt('Appreciation of eyewitness evidence')">
                                <span class="prompt-icon"><i class="fas fa-user-shield"></i></span>
                                <span class="prompt-copy">
                                    <span class="prompt-title">Eyewitness evidence</span>
                                    <span class="prompt-text">Compare how courts assess credibility and contradictions.</span>
                                </span>
                            </button>
                            <button class="prompt-card" type="button" onclick="usePrompt('Sentencing principles for murder')">
                                <span class="prompt-icon"><i class="fas fa-scale-unbalanced"></i></span>
                                <span class="prompt-copy">
                                    <span class="prompt-title">Sentencing principles</span>
                                    <span class="prompt-text">Explore aggravating and mitigating factors in murder sentencing.</span>
                                </span>
                            </button>
                        </div>
                    </div>
                `;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSend();
                }
            }
            
            async function handleSend() {
                const input = document.getElementById('chatInput');
                const sendButton = document.getElementById('sendButton');
                const query = input.value.trim();
                if (!query || sendButton?.disabled) return;

                const welcomeScreen = document.getElementById('welcomeScreen');
                if (welcomeScreen) welcomeScreen.style.display = 'none';

                addMessage(query, 'user');
                input.value = '';
                autoGrowInput();

                if (activeChatCardId) {
                    await handleChatMessage(query, activeChatCardId);
                } else {
                    await handleSearch(query);
                }
            }
            
            async function handleSearch(query) {
                showLoading('Searching legal database...');
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${sessionId}`
                        },
                        body: JSON.stringify({ query, top_k: currentTopK, chat_id: chatId })
                    });

                    const data = await response.json().catch(() => ({}));
                    hideLoading();

                    if (!response.ok) {
                        addMessage(data.detail || 'Search failed. Please try again.', 'assistant');
                        await loadChats();
                        return;
                    }

                    if (data.chat_id) chatId = data.chat_id;
                    
                    if (data.success) {
                        data.judgments.forEach((judgment, idx) => {
                            const cardId = `card_${Date.now()}_${idx}`;
                            judgmentStore[cardId] = {
                                fullText: judgment.full_text,
                                fileName: judgment.file_name,
                                score:    judgment.score
                            };
                            showJudgmentResult(judgment, cardId);
                        });
                        addMessage(`Found ${data.judgments.length} relevant judgment(s) with score ≥ 75%`, 'assistant');
                    } else {
                        addMessage(data.message || 'No strong judgment found. Try refining your query.', 'assistant');
                    }
                    await loadChats();
                } catch (error) {
                    hideLoading();
                    addMessage('Error: ' + error.message, 'assistant');
                    await loadChats();
                }
            }
            
            function showJudgmentResult(judgment, cardId) {
                const rawScore = Number(judgment.score);
                const scoreLabel = Number.isFinite(rawScore) ? `${Math.round(rawScore * 100)}% match` : 'Matched';
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message message-assistant message-with-card';
                messageDiv.innerHTML = `
                    <div class="message-content">
                        <div class="judgment-card">
                            <div class="judgment-header">
                                <div class="judgment-heading">
                                    <div class="judgment-doc-icon"><i class="fas fa-file-lines"></i></div>
                                    <div>
                                        <div class="judgment-kicker">Retrieved judgment</div>
                                        <div class="judgment-file">${escapeHtml(judgment.file_name)}</div>
                                        <div class="judgment-meta">Peshawar High Court · Criminal jurisdiction</div>
                                    </div>
                                </div>
                                <span class="score-badge">
                                    <i class="fas fa-bullseye"></i> ${scoreLabel}
                                </span>
                            </div>
                            <div class="judgment-actions">
                                <button class="action-btn" id="viewBtn_${cardId}" type="button" onclick="viewJudgment('${cardId}')">
                                    <i class="fas fa-eye"></i> View full text
                                </button>
                                <button class="action-btn" id="summaryBtn_${cardId}" type="button" onclick="summarizeJudgment('${cardId}')">
                                    <i class="fas fa-file-lines"></i> Summary
                                </button>
                                <button class="action-btn" id="chatBtn_${cardId}" type="button" onclick="enterChatMode('${cardId}')">
                                    <i class="fas fa-comments"></i> Discuss
                                </button>
                                <button class="action-btn" type="button" onclick="copyCitation('${cardId}')">
                                    <i class="fas fa-copy"></i> Copy citation
                                </button>
                            </div>
                            <div id="judgmentDisplay_${cardId}"></div>
                        </div>
                    </div>
                `;
                document.getElementById('chatMessages').appendChild(messageDiv);
                scrollToBottom();
            }

            function viewJudgment(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;

                const display = document.getElementById('judgmentDisplay_' + cardId);
                const button = document.getElementById('viewBtn_' + cardId);
                const fullTextOpen = display.querySelector('.judgment-text');
                if (fullTextOpen) {
                    display.innerHTML = '';
                    button?.classList.remove('active');
                    if (button) button.innerHTML = '<i class="fas fa-eye"></i> View full text';
                } else {
                    display.innerHTML = `<div class="judgment-text">${escapeHtml(judgment.fullText)}</div>`;
                    button?.classList.add('active');
                    if (button) button.innerHTML = '<i class="fas fa-eye-slash"></i> Hide full text';
                    scrollToBottom();
                }
            }
            
            async function summarizeJudgment(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;

                const display = document.getElementById('judgmentDisplay_' + cardId);
                const button = document.getElementById('summaryBtn_' + cardId);
                const viewButton = document.getElementById('viewBtn_' + cardId);
                if (viewButton) {
                    viewButton.classList.remove('active');
                    viewButton.innerHTML = '<i class="fas fa-eye"></i> View full text';
                }
                if (display) {
                    display.innerHTML = `
                        <div class="summary-placeholder">
                            <i class="fas fa-spinner fa-spin"></i>
                            <span>Reading the judgment and preparing a structured summary...</span>
                        </div>
                    `;
                }
                if (button) button.disabled = true;

                try {
                    const response = await fetch('/api/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${sessionId}`
                        },
                        body: JSON.stringify({ text: judgment.fullText })
                    });

                    const data = await response.json();

                    if (response.ok && data.summary && data.summary.trim()) {
                        judgment.summary = data.summary;
                        display.innerHTML = `
                            <div class="summary-box">
                                <div class="summary-box-header">
                                    <div class="summary-box-title"><i class="fas fa-scale-balanced"></i> AI case summary</div>
                                    <button class="action-btn" type="button" onclick="copySummary('${cardId}')">
                                        <i class="fas fa-copy"></i> Copy
                                    </button>
                                </div>
                                <div class="summary-content">${formatSummary(data.summary)}</div>
                            </div>
                        `;
                        scrollToBottom();
                    } else {
                        display.innerHTML = '';
                        showNotification(data.detail || 'The summary was empty. Please try again.', 'error');
                    }
                } catch (error) {
                    display.innerHTML = '';
                    showNotification('Error generating summary: ' + error.message, 'error');
                } finally {
                    if (button) button.disabled = false;
                }
            }

            async function copySummary(cardId) {
                const summary = judgmentStore[cardId]?.summary;
                if (!summary) return;
                await copyText(summary, 'Summary copied to clipboard');
            }

            async function copyCitation(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;

                const score = Number(judgment.score);
                const scoreLine = Number.isFinite(score) ? ` Relevance: ${Math.round(score * 100)}%.` : '';
                await copyText(`${judgment.fileName} — Peshawar High Court criminal judgment.${scoreLine}`, 'Citation copied to clipboard');
            }

            async function copyText(text, successMessage) {
                try {
                    if (navigator.clipboard?.writeText) {
                        await navigator.clipboard.writeText(text);
                    } else {
                        const area = document.createElement('textarea');
                        area.value = text;
                        area.style.position = 'fixed';
                        area.style.opacity = '0';
                        document.body.appendChild(area);
                        area.select();
                        document.execCommand('copy');
                        area.remove();
                    }
                    showNotification(successMessage, 'success');
                } catch (error) {
                    showNotification('Could not copy to clipboard', 'error');
                }
            }
            
            function enterChatMode(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;
                
                if (activeChatCardId) {
                    const prevBtn = document.getElementById('chatBtn_' + activeChatCardId);
                    if (prevBtn) prevBtn.classList.remove('active');
                }
                
                activeChatCardId = cardId;
                const chatBtn = document.getElementById('chatBtn_' + cardId);
                if (chatBtn) chatBtn.classList.add('active');
                
                document.getElementById('chatInput').placeholder = 'Ask a question about this judgment... (type "exit" to leave chat)';
                autoGrowInput();
                addMessage('Chat mode is active for this judgment. Ask a follow-up question, or type "exit" to return to search.', 'assistant');
            }
            
            async function handleChatMessage(question, cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) {
                    addMessage('Error: Judgment not found.', 'assistant');
                    return;
                }
                
                showLoading('Searching ...');
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${sessionId}`
                        },
                        body: JSON.stringify({
                            question:   question,
                            context:    judgment.fullText,
                            session_id: sessionId,
                            chat_id:    chatId,
                            card_id:    cardId
                        })
                    });
                    
                    const data = await response.json();
                    hideLoading();
                    if (data.chat_id) chatId = data.chat_id;
                    
                    if (data.exit_chat && data.silent) {
                        activeChatCardId = null;
                        document.getElementById('chatInput').placeholder = 'Enter your legal query...';
                        const chatBtn = document.getElementById('chatBtn_' + cardId);
                        if (chatBtn) chatBtn.classList.remove('active');
                    } else if (response.ok) {
                        addMessage(data.answer, 'assistant');
                    } else {
                        addMessage('Error: ' + (data.detail || 'Chat request failed'), 'assistant');
                    }
                    if (response.ok) await loadChats();
                } catch (error) {
                    hideLoading();
                    addMessage('Error: ' + error.message, 'assistant');
                }
            }
            
            // ── Chats / Saved ─────────────────────────────────────────
            async function loadChats() {
                try {
                    const response = await fetch('/api/chats', {
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                    const data        = await response.json();
                    chats             = data.chats || [];
                    const historyList = document.getElementById('historyList');
                    
                    if (chats.length === 0) {
                        historyList.innerHTML = '<div class="sidebar-empty">Your research conversations will appear here.</div>';
                        return;
                    }

                    historyList.innerHTML = chats.slice(0, 30).map(chat => {
                        const date    = new Date(chat.updated_at);
                        const timeStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
                                        date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                        const count   = Math.ceil((chat.message_count || 0) / 2);
                        const active  = chat.id === chatId ? ' active' : '';
                        return `
                            <div class="history-item${active}" role="button" tabindex="0" title="Right-click for options" onclick="openChat('${chat.id}')" onkeydown="if(event.key==='Enter') openChat('${chat.id}')" oncontextmenu="openChatMenu(event, '${chat.id}')">
                                <div class="history-item-query">${escapeHtml(chat.title || 'New chat')}</div>
                                <div class="history-item-time">${timeStr} · ${count} turn${count === 1 ? '' : 's'}</div>
                            </div>
                        `;
                    }).join('');
                } catch (error) {
                    console.error('Error loading chats:', error);
                }
            }

            function openChatMenu(event, id) {
                event.preventDefault();
                event.stopPropagation();

                const menu = document.getElementById('chatContextMenu');
                if (!menu) return;

                contextChatId = id;
                menu.classList.add('open');

                const margin = 8;
                const rect = menu.getBoundingClientRect();
                const left = Math.max(margin, Math.min(event.clientX, window.innerWidth - rect.width - margin));
                const top = Math.max(margin, Math.min(event.clientY, window.innerHeight - rect.height - margin));
                menu.style.left = left + 'px';
                menu.style.top = top + 'px';
            }

            function closeChatMenu() {
                document.getElementById('chatContextMenu')?.classList.remove('open');
                contextChatId = null;
            }

            async function deleteChatFromMenu() {
                const id = contextChatId;
                closeChatMenu();
                if (!id) return;
                if (!confirm('Delete this conversation? This action cannot be undone.')) return;

                try {
                    const response = await fetch('/api/chats/' + encodeURIComponent(id), {
                        method: 'DELETE',
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });

                    if (!response.ok) {
                        const data = await response.json().catch(() => ({}));
                        throw new Error(data.detail || 'Could not delete conversation');
                    }

                    chats = chats.filter(chat => chat.id !== id);
                    if (chatId === id) {
                        await startNewChat();
                    } else {
                        await loadChats();
                    }
                    showNotification('Conversation deleted', 'success');
                } catch (error) {
                    showNotification('Error deleting conversation: ' + error.message, 'error');
                }
            }
            
            async function openChat(id) {
                try {
                    const response = await fetch('/api/chats/' + encodeURIComponent(id), {
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                    if (!response.ok) {
                        showNotification('Could not open chat', 'error');
                        return;
                    }
                    const data = await response.json();
                    const chat = data.chat;
                    
                    chatId           = chat.id;
                    activeChatCardId = null;
                    judgmentStore    = {};
                    
                    const messages = document.getElementById('chatMessages');
                    messages.innerHTML = '';
                    
                    (chat.messages || []).forEach((message, messageIndex) => {
                        if (message.role === 'user') {
                            addMessage(message.content, 'user');
                            return;
                        }
                        if (Array.isArray(message.judgments) && message.judgments.length) {
                            message.judgments.forEach((judgment, index) => {
                                const cardId = `card_${message.id}_${index}`;
                                judgmentStore[cardId] = {
                                    fullText: judgment.full_text,
                                    fileName: judgment.file_name,
                                    score:    judgment.score
                                };
                                showJudgmentResult(judgment, cardId);
                            });
                        }
                        addMessage(message.content, 'assistant');
                    });
                    
                    if (!(chat.messages || []).length) {
                        messages.innerHTML = welcomeMarkup();
                    }
                    document.getElementById('chatInput').placeholder = 'Enter your legal query...';
                    autoGrowInput();
                    closeSidebar();
                    await loadChats();
                    scrollToBottom();
                } catch (error) {
                    showNotification('Error opening chat: ' + error.message, 'error');
                }
            }
            
            async function loadSavedJudgments() {
                try {
                    const response = await fetch('/api/saved', {
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                    const data      = await response.json();
                    const savedList = document.getElementById('savedList');
                    
                    if (data.saved.length === 0) {
                        savedList.innerHTML = '<div class="sidebar-empty">Save a useful judgment to find it here later.</div>';
                        return;
                    }

                    savedList.innerHTML = data.saved.slice(0, 20).map(item => {
                        const score = Number(item.score);
                        return `
                            <div class="history-item">
                                <div class="history-item-query">${escapeHtml(item.file_name)}</div>
                                <div class="history-item-time">${Number.isFinite(score) ? Math.round(score * 100) + '% match' : 'Saved judgment'}</div>
                            </div>
                        `;
                    }).join('');
                } catch (error) {
                    console.error('Error loading saved:', error);
                }
            }
            
            // ── File Upload ───────────────────────────────────────────
            function triggerFileUpload() {
                document.getElementById('fileUpload').click();
            }
            
            async function handleFileUpload(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                showNotification('Uploading and indexing file...', 'success');
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${sessionId}` },
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showNotification('File uploaded successfully!', 'success');
                        addMessage(`Uploaded and indexed: ${data.filename} (${data.text_length} characters)`, 'assistant');
                    } else {
                        showNotification(data.detail || 'Upload failed', 'error');
                    }
                } catch (error) {
                    showNotification('Error uploading file', 'error');
                }
                
                event.target.value = '';
            }
            
            // ── UI Helpers ────────────────────────────────────────────
            function escapeHtml(value) {
                return String(value ?? '').replace(/[&<>"']/g, ch => ({
                    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
                }[ch]));
            }

            function formatSummary(value) {
                const lines = escapeHtml(value ?? '').split(/\\r?\\n/);
                const inline = text => text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
                let html = '';
                let listOpen = false;
                const closeList = () => {
                    if (listOpen) {
                        html += '</ul>';
                        listOpen = false;
                    }
                };

                lines.forEach(rawLine => {
                    const line = rawLine.trim();
                    if (!line) {
                        closeList();
                        return;
                    }

                    const heading = line.match(/^#{1,4}\\s+(.+)$/) || line.match(/^\\*\\*(.+?)\\*\\*:?$/);
                    if (heading) {
                        closeList();
                        html += `<h4>${inline(heading[1])}</h4>`;
                        return;
                    }

                    const bullet = line.match(/^[-*•]\\s+(.+)$/);
                    if (bullet) {
                        if (!listOpen) {
                            html += '<ul>';
                            listOpen = true;
                        }
                        html += `<li>${inline(bullet[1])}</li>`;
                        return;
                    }

                    closeList();
                    html += `<p>${inline(line)}</p>`;
                });

                closeList();
                return html;
            }

            function addMessage(content, role) {
                const messageDiv      = document.createElement('div');
                messageDiv.className  = `message message-${role}`;
                const body            = role === 'assistant' ? formatSummary(content) : escapeHtml(content);
                messageDiv.innerHTML  = `<div class="message-content">${body}</div>`;
                document.getElementById('chatMessages').appendChild(messageDiv);
                scrollToBottom();
            }
            
            function showLoading(text) {
                hideLoading();

                const loadingDiv     = document.createElement('div');
                loadingDiv.id        = 'loadingMessage';
                loadingDiv.className = 'message message-assistant';
                loadingDiv.innerHTML = `
                    <div class="loading-message">
                        <div class="loading-dots">
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                        </div>
                        <span>${escapeHtml(text)}</span>
                        <span class="loading-elapsed" id="loadingElapsed">0s</span>
                    </div>
                `;
                document.getElementById('chatMessages').appendChild(loadingDiv);
                const sendButton = document.getElementById('sendButton');
                if (sendButton) sendButton.disabled = true;

                const startedAt = Date.now();
                clearInterval(loadingTimer);
                loadingTimer = setInterval(() => {
                    const elapsed = document.getElementById('loadingElapsed');
                    if (elapsed) elapsed.textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
                }, 1000);

                scrollToBottom();
            }

            function hideLoading() {
                clearInterval(loadingTimer);
                loadingTimer = null;
                const loading = document.getElementById('loadingMessage');
                if (loading) loading.remove();
                const sendButton = document.getElementById('sendButton');
                if (sendButton) sendButton.disabled = false;
            }

            function scrollToBottom() {
                const chatMessages = document.getElementById('chatMessages');
                if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function showNotification(message, type) {
                const notification      = document.createElement('div');
                notification.className  = `notification ${type}`;
                notification.innerHTML  = `
                    <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                    <span>${escapeHtml(message)}</span>
                `;
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 3200);
            }

            document.addEventListener('keydown', event => {
                if (event.key === 'Escape') {
                    closeSidebar();
                    closeChatMenu();
                }
            });

            document.addEventListener('click', event => {
                const menu = document.getElementById('chatContextMenu');
                if (menu && !menu.contains(event.target)) closeChatMenu();
            });

            window.addEventListener('resize', closeChatMenu);
            window.addEventListener('scroll', closeChatMenu, true);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" makes the server reachable on your local network too
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")