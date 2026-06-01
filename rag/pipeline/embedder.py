# rag/pipeline/embedder.py

from typing import List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

DENSE_MODEL = "all-MiniLM-L6-v2"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
BATCH_SIZE = 32

_dense_model = None
_sparse_model = None

@dataclass
class SparseVector:
    indices : List[int]
    values : List[float]

def get_dense_model():
    global _dense_model
    if _dense_model is None:
        print(f"[EMBEDDER] Loading dense model: {DENSE_MODEL}")
        _dense_model = SentenceTransformer(DENSE_MODEL)
    return _dense_model

def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        print(f"[EMBEDDER] Loading sparse model: {SPARSE_MODEL} ")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model

def embed_chunks(chunks: List) -> List:
    """
    Attach dense AND sparse embeddings to every chunk.
    Dense: 384-dim float vector (semantic similarity)
    Sparse: SPLADE token weights (keyword matching in Qdrant)
    """
    texts = [chunk.text for chunk in chunks]
    total = len(texts)

    # Dense in batches
    print(f"[EMBEDDER] Dense embedding {total} chunks")
    dense_model = get_dense_model()
    all_dense = []
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = dense_model.encode(
            batch, show_progress_bar=True, normalize_embeddings=True
        )
        all_dense.extend(embeddings.tolist())
        print(f"[EMBEDDER] Dense: {min(i + BATCH_SIZE, total)}/ {total}")

    # sparse in batches (fastembed handles internally)
    print(f"[EMBEDDER] sparse embedding {total} chunks")
    sparse_model = get_sparse_model()
    all_sparse = list(sparse_model.embed(texts, batch_size=BATCH_SIZE))

    for chunk, dense, sparse in zip(chunks, all_dense, all_sparse):
        chunk.embedding = dense
        chunk.sparse_embedding = SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )
    print(f"[EMBEDDER] Done - {total} dense + {total} sparse vectors")
    return chunks