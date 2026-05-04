"""
RAG Retriever
─────────────
Retriever connected to Qdrant.
Used by all RAG components.
"""
import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_core.tools import base
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL","http://qdrant:6333")

COLLECTION_NAIVE = "hr_naive"
COLLECTION_POLICY_AWARE = "hr_policy_aware"


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Get embedding model instance.
 
    Cached with lru_cache so the model is loaded once
    and reused across all calls in the same process.
    Without this, every retriever.invoke() reloads the
    model from disk — wastes 2-3 seconds each time.

    First call can take 1–3+ minutes: imports PyTorch/sentence-transformers
    and may download weights. Progress is printed with flush so the terminal
    does not look frozen.
    """
    print(
        "[LOAD] Loading embedding model (PyTorch/sentence-transformers; first run is slow)…",
        flush=True,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True},
    )
    print("[OK] Embedding model loaded", flush=True)
    return embeddings

def get_vector_store(
    collection: str = COLLECTION_NAIVE,
) -> QdrantVectorStore:
    """Get connected Qdrant vector store"""
    embeddings = get_embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection,
    )

def get_retriever(
    collection: str = COLLECTION_NAIVE,
    search_type: str = "similarity",
    k: int = 5,
    ):
    """
    Langchain Retriever by Qdrant

    For baseline: search_type="similarity" (pure relevance)
    For improved - actual : search_type="mmr" (diversity)


    1. Similarity Search
        - This method retrieves documents based on vector similarity.
        - It finds the most similar documents to the query vector based on cosine similarity.
        - Use this when you want to retrieve the top k most similar documents.

    2. Max Marginal Relevance (MMR)
        - This method balances between selecting documents that are relevant to the query and diverse among themselves.
        - 'fetch_k' specifies the number of documents to initially fetch based on similarity.
        - 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
        - Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
        - Note: Relevance measures how closely documents match the query.
        - Note: Diversity ensures that the retrieved documents are not too similar to each other,providing a broader range of information.

    3. Similarity Score Threshold
        - This method retrieves documents that exceed a certain similarity score threshold.
        - 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
        - Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.


    Get a LangChain retriever for a specific collection.
 
    Args:
        collection: Which Qdrant collection to search.
            COLLECTION_NAIVE        → blind chunks
            COLLECTION_POLICY_AWARE → policy-aware chunks
        search_type: Retrieval strategy.
            "similarity" → pure cosine similarity (baseline)
            "mmr"        → maximal marginal relevance (diverse)
        k: Number of documents to retrieve.
    """
    vector_store = get_vector_store(collection=collection)
    return vector_store.as_retriever(
        search_type = search_type,
        search_kwargs={"k":k},
    )


def get_rerank_retriever(
    collection: str = None,
    search_type: str = "mmr",
    initial_k: int = 15,
    final_k: int = 5,
):
    """
    Two-stage retriever: vector search → cross-encoder re-rank.
 
    Stage 1: Retrieve initial_k with vector search (fast, approximate)
    Stage 2: Re-rank with cross-encoder, keep final_k (slow, precise)
 
    Why two stages?
        Cross-encoders are accurate but slow. Running them on all 127
        policies would be too slow. So we use fast vector search to
        narrow to ~15 candidates, then re-rank those 15 precisely.
 
    Args:
        collection: Qdrant collection name
        search_type: "mmr" or "similarity" for stage 1
        initial_k: How many to retrieve in stage 1 (cast wide)
        final_k: How many to keep after re-ranking (filter down)
    """
    # Default to policy-aware if not specified
    if collection is None:
        collection = COLLECTION_POLICY_AWARE

    # Lazy import = flashrank downloads model on first use
    # Keep it inside function means other scripts that don't use re-ranking logic won't trigger the download
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank


    # Stage 1 : broad vector retrieval
    base_retriever = get_retriever(
        collection = collection,
        search_type = search_type,
        k=initial_k,
    )

    # stage 2: cross-encoder re-ranking
    compressor = FlashrankRerank(
        top_n = final_k,
    )

    return ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = base_retriever,
    )





    