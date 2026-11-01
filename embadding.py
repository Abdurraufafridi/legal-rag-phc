# =====================================
# MULTI-GRANULARITY EMBEDDING SCRIPT
# (Chunk + Full Document Embedding)
# Using BGE-M3 Hybrid (Dense + Sparse)
# =====================================

import json
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    VectorParams,
    Distance,
    SparseVectorParams
)
from collections import defaultdict
import uuid

import config as app_config

# -----------------------------
# Configuration
# -----------------------------

EMBEDDING_MODEL_NAME = app_config.EMBEDDING_MODEL_NAME

CHUNKED_DATA_PATH = str(app_config.CHUNKED_JSON)
QDRANT_PATH = str(app_config.QDRANT_PATH)
COLLECTION_NAME = app_config.COLLECTION_NAME
VECTORIZER_PATH = str(app_config.VECTORIZER_PATH)

BATCH_SIZE = app_config.EMBEDDING_BATCH_SIZE
MAX_SEQ_LENGTH = app_config.EMBEDDING_MAX_SEQ_LENGTH  # Prevent attention explosion
QDRANT_URL = app_config.QDRANT_URL
QDRANT_API_KEY = app_config.QDRANT_API_KEY
RECREATE_COLLECTION = app_config.RECREATE_COLLECTION
EMBED_DOCUMENTS = app_config.EMBED_DOCUMENTS

print("=" * 70)
print("MULTI-GRANULARITY LEGAL EMBEDDING PIPELINE")
print("=" * 70)

# -----------------------------
# Step 1: Load Model
# -----------------------------

print("\nLoading BGE-M3 model...")

requested_device = app_config.MODEL_DEVICE
if requested_device == "cuda" and not torch.cuda.is_available():
    requested_device = "cpu"
if requested_device == "mps" and not (
    getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
):
    requested_device = "cpu"
device = requested_device
print(f"Using device: {device}")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=device
)

# Limit maximum token length
embedding_model.max_seq_length = MAX_SEQ_LENGTH

embed_dim = embedding_model.get_sentence_embedding_dimension()
print(f"✓ Embedding dimension: {embed_dim}")

# -----------------------------
# Step 2: Load Data
# -----------------------------

with open(CHUNKED_DATA_PATH, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print(f"✓ Loaded {len(chunked_data)} chunks")

# -----------------------------
# Step 3: Setup Sparse Vectorizer
# -----------------------------

if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, "rb") as f:
        sparse_vectorizer = pickle.load(f)
else:
    sparse_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    print("Fitting sparse vectorizer...")
    sparse_vectorizer.fit([c['text'] for c in chunked_data])

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(sparse_vectorizer, f)

print("✓ Sparse vectorizer ready")

# -----------------------------
# Step 4: Setup Qdrant
# -----------------------------

if QDRANT_URL:
    print(f"Connecting to Qdrant server: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60)
else:
    print(f"Connecting to embedded Qdrant: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)

collections = client.get_collections().collections
existing = [c.name for c in collections]

if RECREATE_COLLECTION and COLLECTION_NAME in existing:
    print(f"Recreating collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)
    existing = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=embed_dim,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    print("✓ Collection created")
else:
    print("✓ Collection exists")

# -----------------------------
# Step 5: Prepare Chunk Points
# -----------------------------

print("\nPreparing chunk-level embeddings...")

chunk_points = []

for chunk in chunked_data:
    text = chunk['text'].strip()
    if not text:
        continue

    chunk_points.append({
        "text": text,
        "file_name": chunk['metadata'].get("file_name", "unknown"),
        "chunk_id": chunk["chunk_id"],
        "metadata": chunk["metadata"],
        "level": "chunk"
    })

print(f"✓ {len(chunk_points)} chunk entries ready")

# -----------------------------
# Step 6: Prepare Document-Level Text
# -----------------------------

print("\nPreparing full-document embeddings...")

documents = defaultdict(list)

for chunk in chunked_data:
    file_name = chunk['metadata'].get("file_name", "unknown")
    documents[file_name].append(chunk['text'])

document_points = []

for file_name, texts in documents.items():
    full_text = "\n".join(texts)

    document_points.append({
        "text": full_text,
        "file_name": file_name,
        "chunk_id": f"doc_{file_name}",
        "metadata": {"file_name": file_name},
        "level": "document"
    })

print(f"✓ {len(document_points)} full documents prepared")

if not EMBED_DOCUMENTS:
    document_points = []

# Merge both
all_points = chunk_points + document_points
print(f"\nTotal embeddings to create: {len(all_points)}")

# -----------------------------
# Step 7: Embed & Upload
# -----------------------------

def create_point(data, dense_vector, sparse_vector):
    sparse_vec = SparseVector(
        indices=sparse_vector.indices.tolist(),
        values=sparse_vector.data.tolist()
    )

    return PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_vector.tolist(),
            "sparse": sparse_vec
        },
        payload={
            "file_name": data["file_name"],
            "text": data["text"],
            "chunk_id": data["chunk_id"],
            "level": data["level"],
            "metadata": data["metadata"]
        }
    )

print("\nEmbedding & Uploading...")

total_uploaded = 0

for i in tqdm(range(0, len(all_points), BATCH_SIZE)):
    batch = all_points[i:i+BATCH_SIZE]
    texts = [b["text"] for b in batch]

    with torch.no_grad():
        dense_vectors = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    sparse_vectors = sparse_vectorizer.transform(texts)

    qdrant_points = []
    for idx, data in enumerate(batch):
        qdrant_points.append(
            create_point(data, dense_vectors[idx], sparse_vectors[idx])
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points,
        wait=False,
    )
    total_uploaded += len(qdrant_points)

    # Clear GPU cache after each batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n✓ Upload complete! {total_uploaded} vectors stored.")

print("\n" + "=" * 70)
print("MULTI-LEVEL EMBEDDING COMPLETE")
print("=" * 70)
print(f"Chunks: {len(chunk_points)}")
print(f"Documents: {len(document_points)}")
print(f"Total stored: {total_uploaded}")
print("✅ READY FOR ADVANCED RETRIEVAL")
print("=" * 70)

# -----------------------------
# Graceful Shutdown (Fix Warning)
# -----------------------------

try:
    client.close()
except Exception:
    pass