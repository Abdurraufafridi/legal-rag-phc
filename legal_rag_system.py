# ============================================================
# COMPLETE LEGAL RAG SYSTEM WITH FULL FEATURES
# FastAPI + Authentication + Chat History + File Upload
# UPDATED: Continuous Conversation Interface (ChatGPT-style)
# ============================================================

import os
import re
import pickle
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion, PointStruct
from openai import OpenAI

import PyPDF2
from docx import Document

import config as app_config

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = app_config.EMBEDDING_MODEL_NAME
CROSS_ENCODER_NAME = app_config.CROSS_ENCODER_NAME

QDRANT_PATH = str(app_config.QDRANT_PATH)
QDRANT_URL = app_config.QDRANT_URL
QDRANT_API_KEY = app_config.QDRANT_API_KEY
COLLECTION_NAME = app_config.COLLECTION_NAME

VECTORIZER_PATH = str(app_config.VECTORIZER_PATH)
ORIGINAL_TEXT_FOLDER = str(app_config.TEXT_DIRECTORY)

TOP_K = 5
THRESHOLD = 0.60

OPENROUTER_API_KEY = app_config.OPENROUTER_API_KEY

# Simple in-memory storage (replace with database in production)
users_db = {}
sessions_db = {}
chat_history_db = defaultdict(list)
saved_judgments_db = defaultdict(list)

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

class ChatRequest(BaseModel):
    question: str
    context: str
    session_id: str
    card_id: str  # To identify which judgment the chat is about

class SaveJudgmentRequest(BaseModel):
    file_name: str
    text: str
    score: float

class SearchHistoryItem(BaseModel):
    id: str
    query: str
    timestamp: str
    result_file: Optional[str] = None
    score: Optional[float] = None

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Legal RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============================================================
# GLOBAL MODELS
# ============================================================

embedding_model = None
cross_encoder = None
sparse_vectorizer = None
client = None
llm_client = None

@app.on_event("startup")
async def load_models():
    global embedding_model, cross_encoder, sparse_vectorizer, client, llm_client
    
    print("🚀 Loading models...")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, max_length=512)
    
    with open(VECTORIZER_PATH, "rb") as f:
        sparse_vectorizer = pickle.load(f)
    
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60)
    else:
        client = QdrantClient(path=QDRANT_PATH)
    
    llm_client = OpenAI(
        base_url=app_config.OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    print("✅ All models loaded!")

# ============================================================
# AUTHENTICATION FUNCTIONS
# ============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_session(email: str) -> str:
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "email": email,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }
    return session_id

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    if token not in sessions_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    session = sessions_db[token]
    
    if datetime.now() > session["expires_at"]:
        del sessions_db[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )
    
    return session["email"]

# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def hybrid_search(query, limit=50):
    dense_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()
    
    sparse_query = sparse_vectorizer.transform([query])
    
    sparse_vector = SparseVector(
        indices=sparse_query[0].indices.tolist(),
        values=sparse_query[0].data.tolist()
    )
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=limit),
            Prefetch(query=sparse_vector, using="sparse", limit=limit)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True
    )
    
    return results.points

def aggregate_by_document(points):
    documents = defaultdict(list)
    
    for p in points:
        payload = p.payload
        file_name = payload.get("file_name", "unknown")
        documents[file_name].append(payload)
    
    return documents

def rerank_documents(query, documents_dict):
    doc_scores = {}
    
    for file_name, chunks in documents_dict.items():
        texts = [chunk["text"] for chunk in chunks]
        pairs = [[query, text] for text in texts]
        
        scores = cross_encoder.predict(pairs)
        normalized_scores = 1 / (1 + np.exp(-scores))
        best_score = float(np.max(normalized_scores))
        
        doc_scores[file_name] = best_score
    
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_docs[:TOP_K]

def load_full_document(file_name):
    file_path = os.path.join(ORIGINAL_TEXT_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by",
    "with", "about", "regarding", "related", "concerning", "where", "which",
    "that", "this", "these", "those", "is", "are", "was", "were", "be",
    "been", "being", "give", "show", "find", "list", "get", "display",
    "tell", "me", "i", "we", "you", "please", "can", "could", "would",
    "should", "do", "does", "did", "want", "need", "all", "any", "some",
    "relevant", "judgment", "judgments", "case", "cases", "decision",
    "decisions", "ruling", "rulings", "document", "documents", "record",
    "records",
}

def normalize_search_query(query):
    """Reduce conversational queries to their key search terms."""
    raw = query.strip()
    if not raw or ".txt" in raw.lower():
        return raw
    cleaned = re.sub(r"[^\w\s\-/]", " ", raw.lower())
    words = [w for w in cleaned.split() if w not in _QUERY_STOPWORDS]
    return " ".join(words).strip() or raw

def retrieve_full_judgment(query):
    search_query = normalize_search_query(query)
    results = hybrid_search(search_query)
    
    if not results:
        return None, None, None
    
    documents_dict = aggregate_by_document(results)
    ranked_docs = rerank_documents(search_query, documents_dict)
    
    if not ranked_docs:
        return None, None, None
    
    best_file, best_score = ranked_docs[0]
    
    if best_score < THRESHOLD:
        return None, None, None
    
    full_text = load_full_document(best_file)
    
    return full_text, best_score, best_file

# ============================================================
# FILE PROCESSING
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
    import io
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    import io
    doc_file = io.BytesIO(file_content)
    doc = Document(doc_file)
    
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    return text

# ============================================================
# API ENDPOINTS - AUTHENTICATION
# ============================================================

@app.post("/api/register")
async def register(user: UserRegister):
    if user.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    users_db[user.email] = {
        "email": user.email,
        "password": hash_password(user.password),
        "full_name": user.full_name,
        "created_at": datetime.now().isoformat()
    }
    
    return {"message": "Registration successful", "email": user.email}

@app.post("/api/login")
async def login(user: UserLogin):
    if user.email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    stored_user = users_db[user.email]
    
    if hash_password(user.password) != stored_user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    session_id = create_session(user.email)
    
    return {
        "session_id": session_id,
        "email": user.email,
        "full_name": stored_user["full_name"]
    }

@app.post("/api/logout")
async def logout(email: str = Depends(get_current_user)):
    # Remove all sessions for this user
    to_remove = [sid for sid, sess in sessions_db.items() if sess["email"] == email]
    for sid in to_remove:
        del sessions_db[sid]
    
    return {"message": "Logged out successfully"}

@app.get("/api/profile")
async def get_profile(email: str = Depends(get_current_user)):
    user = users_db.get(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "email": user["email"],
        "full_name": user["full_name"],
        "created_at": user["created_at"]
    }

# ============================================================
# API ENDPOINTS - SEARCH & RAG (UPDATED WITH GROUNDED QA)
# ============================================================

@app.post("/api/search")
async def search_judgment(request: SearchRequest, email: str = Depends(get_current_user)):
    try:
        full_text, score, file_name = retrieve_full_judgment(request.query)
        
        # Save to history
        history_item = {
            "id": str(uuid.uuid4()),
            "query": request.query,
            "timestamp": datetime.now().isoformat(),
            "result_file": file_name if full_text else None,
            "score": score if full_text else None
        }
        chat_history_db[email].append(history_item)
        
        if full_text:
            return {
                "success": True,
                "score": score,
                "full_text": full_text,
                "file_name": file_name,
                "history_id": history_item["id"],
                "card_id": str(uuid.uuid4())  # Unique ID for this judgment card
            }
        else:
            return {
                "success": False,
                "message": "No strong judgment found. Try refining your query."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_judgment(request: dict, email: str = Depends(get_current_user)):
    try:
        text = request.get("text", "")
        
        # UPDATED PROMPT: Grounded summarization
        prompt = f"""You are a legal assistant. Your task is to summarize the judgment provided below.

STRICT RULES:
1. Use ONLY the information present in the judgment text below. Do NOT add any external legal knowledge.
2. Be specific: mention exact parties, dates, courts, sections, and outcomes found in the text.
3. Do NOT provide general legal explanations or background information.
4. If a detail is not present in the text, do not invent or assume it.
5. Structure your summary as: (a) Parties involved, (b) Key facts, (c) Legal issues, (d) Decision/Outcome.

JUDGMENT TEXT:
{text[:3000]}

Provide a concise, grounded summary based solely on the above text."""
        
        response = llm_client.chat.completions.create(
            model=app_config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )
        
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_judgment(request: ChatRequest, email: str = Depends(get_current_user)):
    try:
        # Check for exit commands - UPDATED: Return silent exit
        if request.question.lower().strip() in ["exit", "quit", "/exit", "/quit"]:
            return {"answer": None, "exit_chat": True, "silent": True, "card_id": request.card_id}
        
        # UPDATED PROMPT: Grounded question answering
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
{request.context[:3000]}

QUESTION:
{request.question}

Answer strictly from the judgment text above:"""
        
        response = llm_client.chat.completions.create(
            model=app_config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        
        return {
            "answer": response.choices[0].message.content, 
            "exit_chat": False, 
            "silent": False,
            "card_id": request.card_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# API ENDPOINTS - HISTORY & SAVED
# ============================================================

@app.get("/api/history")
async def get_history(email: str = Depends(get_current_user)):
    history = chat_history_db.get(email, [])
    return {"history": list(reversed(history[-50:]))}  # Last 50 items, newest first

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
        "saved_at": datetime.now().isoformat()
    }
    
    saved_judgments_db[email].append(saved_item)
    
    return {"message": "Judgment saved successfully", "id": saved_item["id"]}

# ============================================================
# API ENDPOINTS - FILE UPLOAD
# ============================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), email: str = Depends(get_current_user)):
    try:
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(content)
        elif file.filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Create embeddings and add to Qdrant
        dense_vector = embedding_model.encode(
            text[:1000],  # Limit text length
            normalize_embeddings=True
        ).tolist()
        
        sparse_query = sparse_vectorizer.transform([text[:1000]])
        sparse_vector = SparseVector(
            indices=sparse_query[0].indices.tolist(),
            values=sparse_query[0].data.tolist()
        )
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector
            },
            payload={
                "file_name": file.filename,
                "text": text,
                "chunk_id": f"uploaded_{file.filename}",
                "level": "document",
                "uploaded_by": email,
                "metadata": {"file_name": file.filename}
            }
        )
        
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        
        return {
            "message": "File uploaded and indexed successfully",
            "filename": file.filename,
            "text_length": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MAIN HTML INTERFACE (UPDATED - CONTINUOUS CONVERSATION)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>⚖️ Legal RAG System</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: #f5f5f5;
                height: 100vh;
                overflow: hidden;
            }
            
            /* Auth Screen */
            .auth-screen {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .auth-container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 400px;
                max-width: 90%;
            }
            
            .auth-header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .auth-header h1 {
                font-size: 2em;
                color: #333;
                margin-bottom: 5px;
            }
            
            .auth-header p {
                color: #666;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
            }
            
            .form-group input {
                width: 100%;
                padding: 12px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 1em;
                transition: all 0.3s;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .btn-auth {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s;
            }
            
            .btn-auth:hover {
                transform: translateY(-2px);
            }
            
            .auth-switch {
                text-align: center;
                margin-top: 20px;
                color: #666;
            }
            
            .auth-switch a {
                color: #667eea;
                cursor: pointer;
                text-decoration: none;
                font-weight: 600;
            }
            
            /* Main App Layout */
            .app-container {
                display: none;
                height: 100vh;
                overflow: hidden;
            }
            
            .app-layout {
                display: flex;
                height: 100%;
            }
            
            /* Sidebar */
            .sidebar {
                width: 280px;
                background: #1a1a1a;
                color: white;
                display: flex;
                flex-direction: column;
                transition: all 0.3s;
            }
            
            .sidebar-header {
                padding: 20px;
                border-bottom: 1px solid #333;
            }
            
            .sidebar-header h2 {
                font-size: 1.2em;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .new-chat-btn {
                width: 100%;
                padding: 12px;
                background: #2d2d2d;
                border: 1px solid #444;
                color: white;
                border-radius: 8px;
                cursor: pointer;
                margin-top: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                transition: all 0.2s;
            }
            
            .new-chat-btn:hover {
                background: #3d3d3d;
            }
            
            .sidebar-content {
                flex: 1;
                overflow-y: auto;
                padding: 15px;
            }
            
            .sidebar-section {
                margin-bottom: 25px;
            }
            
            .sidebar-section h3 {
                font-size: 0.85em;
                color: #888;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .history-item {
                padding: 10px;
                background: #2d2d2d;
                border-radius: 6px;
                margin-bottom: 8px;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.9em;
            }
            
            .history-item:hover {
                background: #3d3d3d;
            }
            
            .history-item-query {
                color: white;
                margin-bottom: 4px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            
            .history-item-time {
                color: #888;
                font-size: 0.8em;
            }
            
            .sidebar-footer {
                padding: 15px;
                border-top: 1px solid #333;
            }
            
            .user-profile {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 8px;
                cursor: pointer;
            }
            
            .user-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 1.1em;
            }
            
            .user-info {
                flex: 1;
            }
            
            .user-name {
                font-weight: 600;
                font-size: 0.9em;
            }
            
            .user-email {
                font-size: 0.75em;
                color: #888;
            }
            
            /* Main Content */
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
            }
            
            .chat-header {
                padding: 20px;
                background: white;
                border-bottom: 1px solid #e5e5e5;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .chat-title {
                font-size: 1.3em;
                font-weight: 600;
                color: #333;
            }
            
            .header-actions {
                display: flex;
                gap: 10px;
            }
            
            .icon-btn {
                width: 36px;
                height: 36px;
                border: none;
                background: #f5f5f5;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .icon-btn:hover {
                background: #e5e5e5;
            }
            
            /* Chat Messages */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
            }
            
            .welcome-screen {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                padding: 40px;
            }
            
            .welcome-screen h2 {
                font-size: 2.5em;
                margin-bottom: 15px;
                color: #333;
            }
            
            .welcome-screen p {
                color: #666;
                font-size: 1.1em;
                max-width: 600px;
            }
            
            .message {
                margin-bottom: 20px;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message-user {
                align-self: flex-end;
                max-width: 80%;
            }
            
            .message-assistant {
                align-self: flex-start;
                max-width: 100%;
            }
            
            .message-content {
                padding: 15px 20px;
                border-radius: 12px;
                line-height: 1.6;
            }
            
            .message-user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .message-assistant .message-content {
                background: #f5f5f5;
                color: #333;
            }
            
            /* Conversation Divider */
            .conversation-divider {
                margin: 30px 0;
                border-top: 2px solid #e5e5e5;
                position: relative;
            }
            
            .conversation-divider::after {
                content: '• • •';
                position: absolute;
                top: -10px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                padding: 0 15px;
                color: #999;
                font-size: 0.9em;
            }
            
            /* Judgment Card */
            .judgment-card {
                background: white;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
            }
            
            .judgment-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .score-badge {
                background: #10b981;
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
            }
            
            .judgment-actions {
                display: flex;
                gap: 10px;
                margin-top: 15px;
                flex-wrap: wrap;
            }
            
            .action-btn {
                padding: 10px 20px;
                border: 1px solid #e5e5e5;
                background: white;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.2s;
                font-size: 0.9em;
            }
            
            .action-btn:hover {
                background: #f5f5f5;
                border-color: #667eea;
                color: #667eea;
            }
            
            .action-btn.active {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            
            .judgment-text {
                background: #f9fafb;
                padding: 15px;
                border-radius: 8px;
                max-height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 0.85em;
                line-height: 1.6;
                margin-top: 15px;
                border: 1px solid #e5e5e5;
            }
            
            .summary-box {
                background: #fffbeb;
                border-left: 4px solid #f59e0b;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
            }
            
            /* Chat Input */
            .chat-input-container {
                padding: 20px;
                border-top: 1px solid #e5e5e5;
                background: white;
            }
            
            .chat-input-wrapper {
                max-width: 900px;
                margin: 0 auto;
                position: relative;
            }
            
            .chat-input {
                width: 100%;
                padding: 15px 110px 15px 50px;
                border: 2px solid #e5e5e5;
                border-radius: 12px;
                font-size: 1em;
                resize: none;
                font-family: 'Inter', sans-serif;
                transition: all 0.3s;
            }
            
            .chat-input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .input-upload-btn {
                position: absolute;
                left: 15px;
                top: 50%;
                transform: translateY(-50%);
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1.2em;
                color: #666;
                transition: all 0.2s;
            }
            
            .input-upload-btn:hover {
                color: #667eea;
            }
            
            .input-send-btn {
                position: absolute;
                right: 15px;
                top: 50%;
                transform: translateY(-50%);
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                color: white;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
            }
            
            .input-send-btn:hover {
                transform: translateY(-50%) scale(1.05);
            }
            
            .input-send-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            /* Loading */
            .loading-message {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 12px;
                max-width: fit-content;
            }
            
            .loading-dots {
                display: flex;
                gap: 4px;
            }
            
            .loading-dot {
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out;
            }
            
            .loading-dot:nth-child(1) { animation-delay: -0.32s; }
            .loading-dot:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            
            /* Notifications */
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                animation: slideInRight 0.3s ease;
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            .notification.success {
                border-left: 4px solid #10b981;
            }
            
            .notification.error {
                border-left: 4px solid #ef4444;
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .sidebar {
                    position: fixed;
                    left: -280px;
                    height: 100vh;
                    z-index: 100;
                }
                
                .sidebar.open {
                    left: 0;
                }
            }
            
            .hidden {
                display: none !important;
            }
        </style>
    </head>
    <body>
        <!-- Authentication Screen -->
        <div id="authScreen" class="auth-screen">
            <div class="auth-container">
                <!-- Login Form -->
                <div id="loginForm">
                    <div class="auth-header">
                        <h1>⚖️ Legal RAG</h1>
                        <p>Sign in to continue</p>
                    </div>
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="loginEmail" placeholder="your@email.com">
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="loginPassword" placeholder="••••••••">
                    </div>
                    <button class="btn-auth" onclick="handleLogin()">Sign In</button>
                    <div class="auth-switch">
                        Don't have an account? <a onclick="showRegister()">Sign up</a>
                    </div>
                </div>
                
                <!-- Register Form -->
                <div id="registerForm" class="hidden">
                    <div class="auth-header">
                        <h1>⚖️ Legal RAG</h1>
                        <p>Create your account</p>
                    </div>
                    <div class="form-group">
                        <label>Full Name</label>
                        <input type="text" id="registerName" placeholder="John Doe">
                    </div>
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="registerEmail" placeholder="your@email.com">
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="registerPassword" placeholder="••••••••">
                    </div>
                    <button class="btn-auth" onclick="handleRegister()">Sign Up</button>
                    <div class="auth-switch">
                        Already have an account? <a onclick="showLogin()">Sign in</a>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Application -->
        <div id="appContainer" class="app-container">
            <div class="app-layout">
                <!-- Sidebar -->
                <div class="sidebar" id="sidebar">
                    <div class="sidebar-header">
                        <h2><i class="fas fa-balance-scale"></i> Legal RAG</h2>
                        <button class="new-chat-btn" onclick="startNewChat()">
                            <i class="fas fa-plus"></i> New Search
                        </button>
                    </div>
                    
                    <div class="sidebar-content">
                        <div class="sidebar-section">
                            <h3>Search History</h3>
                            <div id="historyList"></div>
                        </div>
                        
                        <div class="sidebar-section">
                            <h3>Saved Judgments</h3>
                            <div id="savedList"></div>
                        </div>
                    </div>
                    
                    <div class="sidebar-footer">
                        <div class="user-profile" onclick="handleLogout()">
                            <div class="user-avatar" id="userAvatar">U</div>
                            <div class="user-info">
                                <div class="user-name" id="userName">User</div>
                                <div class="user-email" id="userEmail">user@email.com</div>
                            </div>
                            <i class="fas fa-sign-out-alt"></i>
                        </div>
                    </div>
                </div>
                
                <!-- Main Content -->
                <div class="main-content">
                    <div class="chat-header">
                        <div class="chat-title">Pakistani Legal Judgments</div>
                        <div class="header-actions">
                            <button class="icon-btn" onclick="loadHistory()">
                                <i class="fas fa-history"></i>
                            </button>
                            <button class="icon-btn" onclick="loadSavedJudgments()">
                                <i class="fas fa-bookmark"></i>
                            </button>
                        </div>
                    </div>
                    
                    <div class="chat-messages" id="chatMessages">
                        <div class="welcome-screen" id="welcomeScreen">
                            <h2>⚖️ Pakistani Legal RAG System</h2>
                            <p>Search and analyze Pakistani legal judgments with AI-powered retrieval and chat capabilities</p>
                        </div>
                    </div>
                    
                    <div class="chat-input-container">
                        <div class="chat-input-wrapper">
                            <button class="input-upload-btn" onclick="triggerFileUpload()">
                                <i class="fas fa-paperclip"></i>
                            </button>
                            <input type="file" id="fileUpload" class="hidden" accept=".pdf,.txt,.docx" onchange="handleFileUpload(event)">
                            <textarea 
                                id="chatInput" 
                                class="chat-input" 
                                placeholder="Enter your legal query..."
                                rows="1"
                                onkeypress="handleKeyPress(event)"
                            ></textarea>
                            <button class="input-send-btn" onclick="handleSend()">
                                Send <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Global State - UPDATED for continuous conversation
            let sessionId = null;
            let currentUser = null;
            let judgmentStore = {};  // Store all judgments by cardId
            let activeChatCardId = null;  // Track which judgment is being chatted with
            
            // Auth Functions
            function showRegister() {
                document.getElementById('loginForm').classList.add('hidden');
                document.getElementById('registerForm').classList.remove('hidden');
            }
            
            function showLogin() {
                document.getElementById('registerForm').classList.add('hidden');
                document.getElementById('loginForm').classList.remove('hidden');
            }
            
            async function handleRegister() {
                const name = document.getElementById('registerName').value;
                const email = document.getElementById('registerEmail').value;
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
                const email = document.getElementById('loginEmail').value;
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
                        sessionId = data.session_id;
                        currentUser = data;
                        
                        // Update UI
                        document.getElementById('authScreen').style.display = 'none';
                        document.getElementById('appContainer').style.display = 'block';
                        document.getElementById('userName').textContent = data.full_name;
                        document.getElementById('userEmail').textContent = data.email;
                        document.getElementById('userAvatar').textContent = data.full_name.charAt(0).toUpperCase();
                        
                        loadHistory();
                        showNotification('Welcome back!', 'success');
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
                
                sessionId = null;
                currentUser = null;
                document.getElementById('appContainer').style.display = 'none';
                document.getElementById('authScreen').style.display = 'flex';
                showNotification('Logged out successfully', 'success');
            }
            
            // Chat Functions - UPDATED for continuous conversation
            function startNewChat() {
                activeChatCardId = null;
                judgmentStore = {};
                
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = `
                    <div class="welcome-screen">
                        <h2>⚖️ Pakistani Legal RAG System</h2>
                        <p>Search and analyze Pakistani legal judgments with AI-powered retrieval and chat capabilities</p>
                    </div>
                `;
                
                document.getElementById('chatInput').value = '';
                document.getElementById('chatInput').placeholder = 'Enter your legal query...';
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSend();
                }
            }
            
            async function handleSend() {
                const input = document.getElementById('chatInput');
                const query = input.value.trim();
                
                if (!query) return;
                
                // Hide welcome screen
                const welcomeScreen = document.getElementById('welcomeScreen');
                if (welcomeScreen) welcomeScreen.style.display = 'none';
                
                // Add user message
                addMessage(query, 'user');
                input.value = '';
                
                // Check if in chat mode for a specific judgment
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
                        body: JSON.stringify({ query })
                    });
                    
                    const data = await response.json();
                    hideLoading();
                    
                    if (data.success) {
                        // Store judgment data
                        judgmentStore[data.card_id] = {
                            fullText: data.full_text,
                            fileName: data.file_name,
                            score: data.score
                        };
                        
                        // Show judgment result (appended, not replaced)
                        showJudgmentResult(data);
                        loadHistory();
                    } else {
                        addMessage(data.message || 'No judgment found. Please try a different query.', 'assistant');
                    }
                } catch (error) {
                    hideLoading();
                    addMessage('Error: ' + error.message, 'assistant');
                }
            }
            
            function showJudgmentResult(data) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message message-assistant';
                messageDiv.innerHTML = `
                    <div class="message-content">
                        <div class="judgment-card">
                            <div class="judgment-header">
                                <strong>✅ Judgment Found</strong>
                                <span class="score-badge">
                                    <i class="fas fa-check-circle"></i> Score: ${data.score.toFixed(2)}
                                </span>
                            </div>
                            <div><strong>File:</strong> ${data.file_name}</div>
                            <div class="judgment-actions">
                                <button class="action-btn" onclick="viewJudgment('${data.card_id}')">
                                    <i class="fas fa-eye"></i> View Full Text
                                </button>
                                <button class="action-btn" onclick="summarizeJudgment('${data.card_id}')">
                                    <i class="fas fa-file-alt"></i> Summary
                                </button>
                                <button class="action-btn" id="chatBtn_${data.card_id}" onclick="enterChatMode('${data.card_id}')">
                                    <i class="fas fa-comments"></i> Chat
                                </button>
                                <button class="action-btn" onclick="saveCurrentJudgment('${data.card_id}')">
                                    <i class="fas fa-bookmark"></i> Save
                                </button>
                            </div>
                            <div id="judgmentDisplay_${data.card_id}"></div>
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
                if (display.innerHTML) {
                    display.innerHTML = '';
                } else {
                    display.innerHTML = `<div class="judgment-text">${judgment.fullText}</div>`;
                    scrollToBottom();
                }
            }
            
            async function summarizeJudgment(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;
                
                showLoading('Generating summary...');
                
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
                    hideLoading();
                    
                    const display = document.getElementById('judgmentDisplay_' + cardId);
                    display.innerHTML = `
                        <div class="summary-box">
                            <strong><i class="fas fa-file-alt"></i> Summary</strong>
                            <p style="margin-top: 10px;">${data.summary}</p>
                        </div>
                    `;
                    scrollToBottom();
                } catch (error) {
                    hideLoading();
                    showNotification('Error generating summary', 'error');
                }
            }
            
            function enterChatMode(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;
                
                // Deactivate previous chat button if any
                if (activeChatCardId) {
                    const prevBtn = document.getElementById('chatBtn_' + activeChatCardId);
                    if (prevBtn) prevBtn.classList.remove('active');
                }
                
                // Activate current chat
                activeChatCardId = cardId;
                const chatBtn = document.getElementById('chatBtn_' + cardId);
                if (chatBtn) chatBtn.classList.add('active');
                
                document.getElementById('chatInput').placeholder = 'Ask a question about this judgment... (type "exit" to leave chat)';
                addMessage('💬 Chat mode activated! Ask me anything about this judgment. Type "exit" to return to search.', 'assistant');
            }
            
            async function handleChatMessage(question, cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) {
                    addMessage('Error: Judgment not found.', 'assistant');
                    return;
                }
                
                showLoading('Thinking...');
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${sessionId}`
                        },
                        body: JSON.stringify({
                            question: question,
                            context: judgment.fullText,
                            session_id: sessionId,
                            card_id: cardId
                        })
                    });
                    
                    const data = await response.json();
                    hideLoading();
                    
                    // UPDATED: Check for silent exit (no message shown)
                    if (data.exit_chat && data.silent) {
                        activeChatCardId = null;
                        document.getElementById('chatInput').placeholder = 'Enter your legal query...';
                        
                        // Deactivate chat button
                        const chatBtn = document.getElementById('chatBtn_' + cardId);
                        if (chatBtn) chatBtn.classList.remove('active');
                        // No message shown to user
                    } else {
                        addMessage(data.answer, 'assistant');
                    }
                } catch (error) {
                    hideLoading();
                    addMessage('Error: ' + error.message, 'assistant');
                }
            }
            
            // History Functions
            async function loadHistory() {
                try {
                    const response = await fetch('/api/history', {
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                    
                    const data = await response.json();
                    const historyList = document.getElementById('historyList');
                    
                    if (data.history.length === 0) {
                        historyList.innerHTML = '<div style="color: #888; font-size: 0.85em; padding: 10px;">No history yet</div>';
                        return;
                    }
                    
                    historyList.innerHTML = data.history.slice(0, 20).map(item => {
                        const date = new Date(item.timestamp);
                        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                        
                        return `
                            <div class="history-item" onclick='searchFromHistory("${item.query.replace(/'/g, "\\'")}�")'>
                                <div class="history-item-query">${item.query}</div>
                                <div class="history-item-time">${timeStr}</div>
                            </div>
                        `;
                    }).join('');
                } catch (error) {
                    console.error('Error loading history:', error);
                }
            }
            
            function searchFromHistory(query) {
                document.getElementById('chatInput').value = query;
                handleSend();
            }
            
            async function loadSavedJudgments() {
                try {
                    const response = await fetch('/api/saved', {
                        headers: { 'Authorization': `Bearer ${sessionId}` }
                    });
                    
                    const data = await response.json();
                    const savedList = document.getElementById('savedList');
                    
                    if (data.saved.length === 0) {
                        savedList.innerHTML = '<div style="color: #888; font-size: 0.85em; padding: 10px;">No saved items</div>';
                        return;
                    }
                    
                    savedList.innerHTML = data.saved.slice(0, 20).map(item => `
                        <div class="history-item">
                            <div class="history-item-query">${item.file_name}</div>
                            <div class="history-item-time">Score: ${item.score.toFixed(2)}</div>
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('Error loading saved:', error);
                }
            }
            
            async function saveCurrentJudgment(cardId) {
                const judgment = judgmentStore[cardId];
                if (!judgment) return;
                
                try {
                    const response = await fetch('/api/save-judgment', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${sessionId}`
                        },
                        body: JSON.stringify({
                            file_name: judgment.fileName,
                            text: judgment.fullText,
                            score: judgment.score
                        })
                    });
                    
                    if (response.ok) {
                        showNotification('Judgment saved!', 'success');
                        loadSavedJudgments();
                    }
                } catch (error) {
                    showNotification('Error saving judgment', 'error');
                }
            }
            
            // File Upload
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
                        headers: {
                            'Authorization': `Bearer ${sessionId}`
                        },
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
            
            // UI Helpers
            function addMessage(content, role) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message message-${role}`;
                messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
                
                document.getElementById('chatMessages').appendChild(messageDiv);
                scrollToBottom();
            }
            
            function showLoading(text) {
                const loadingDiv = document.createElement('div');
                loadingDiv.id = 'loadingMessage';
                loadingDiv.className = 'message message-assistant';
                loadingDiv.innerHTML = `
                    <div class="loading-message">
                        <div class="loading-dots">
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                        </div>
                        <span>${text}</span>
                    </div>
                `;
                
                document.getElementById('chatMessages').appendChild(loadingDiv);
                scrollToBottom();
            }
            
            function hideLoading() {
                const loading = document.getElementById('loadingMessage');
                if (loading) loading.remove();
            }
            
            function scrollToBottom() {
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function showNotification(message, type) {
                const notification = document.createElement('div');
                notification.className = `notification ${type}`;
                notification.innerHTML = `
                    <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                    ${message}
                `;
                
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    notification.remove();
                }, 3000);
            }
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
    uvicorn.run(app, host="0.0.0.0", port=8000)