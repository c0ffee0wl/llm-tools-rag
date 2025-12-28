"""
Cross-encoder reranking for improved relevance using FlashRank.

FlashRank uses ONNX runtime instead of PyTorch, making it lightweight (~4-150MB)
compared to sentence-transformers (~3GB with CUDA dependencies).

Available models:
- ms-marco-TinyBERT-L-2-v2: ~4MB, ultra-fast (default)
- ms-marco-MiniLM-L-12-v2: ~34MB, best quality for English
- ms-marco-MultiBERT-L-12: ~150MB, multilingual (100+ languages)
- rank-T5-flan: ~110MB, best zero-shot performance
"""

from typing import List, Optional

# Lazy-loaded ranker instance
_ranker = None
_ranker_model_name = None


def _get_ranker(model_name: str, max_length: int = 512):
    """
    Get or create FlashRank Ranker instance (lazy-loaded).

    The model is downloaded on first use and cached locally.

    Args:
        model_name: FlashRank model name (e.g., "ms-marco-MultiBERT-L-12")
        max_length: Maximum input length for the model

    Returns:
        FlashRank Ranker instance
    """
    global _ranker, _ranker_model_name

    if _ranker is None or _ranker_model_name != model_name:
        from flashrank import Ranker
        _ranker = Ranker(model_name=model_name, max_length=max_length)
        _ranker_model_name = model_name

    return _ranker


def rerank(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "ms-marco-MultiBERT-L-12",
    top_k: Optional[int] = None
) -> List[str]:
    """
    Rerank documents using cross-encoder relevance scores.

    Uses FlashRank's ONNX-based cross-encoders for fast CPU inference
    without heavy PyTorch/CUDA dependencies.

    Args:
        query: The search query
        documents: List of document texts to rerank
        ids: List of document IDs corresponding to documents
        model_name: FlashRank model name:
            - "ms-marco-TinyBERT-L-2-v2": ~4MB, fastest
            - "ms-marco-MiniLM-L-12-v2": ~34MB, best English
            - "ms-marco-MultiBERT-L-12": ~150MB, multilingual (default)
        top_k: Number of top results to return (None = return all)

    Returns:
        List of document IDs reordered by relevance score (highest first)

    Example:
        >>> ids = rerank(
        ...     query="What is Python?",
        ...     documents=["Python is a snake", "Python is a programming language"],
        ...     ids=["doc1", "doc2"]
        ... )
        >>> ids[0]  # Most relevant document
        'doc2'
    """
    # Validate inputs
    if not documents:
        return ids[:top_k] if top_k else ids

    if len(documents) != len(ids):
        # Mismatch - return original order
        return ids[:top_k] if top_k else ids

    # Get ranker (lazy-loaded)
    ranker = _get_ranker(model_name)

    # Create passages in FlashRank format
    from flashrank import RerankRequest
    passages = [
        {"id": doc_id, "text": doc_text}
        for doc_id, doc_text in zip(ids, documents)
    ]

    # Rerank
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Extract reordered IDs
    result_ids = [r["id"] for r in results]

    # Apply top_k limit if specified
    return result_ids[:top_k] if top_k else result_ids


def rerank_with_scores(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "ms-marco-MultiBERT-L-12",
    top_k: Optional[int] = None
) -> List[tuple]:
    """
    Rerank documents and return scores along with IDs.

    Same as rerank() but returns (id, score) tuples for debugging/analysis.

    Args:
        query: The search query
        documents: List of document texts to rerank
        ids: List of document IDs corresponding to documents
        model_name: FlashRank model name
        top_k: Number of top results to return (None = return all)

    Returns:
        List of (doc_id, score) tuples sorted by score (highest first)
    """
    if not documents or len(documents) != len(ids):
        return [(doc_id, 0.0) for doc_id in (ids[:top_k] if top_k else ids)]

    ranker = _get_ranker(model_name)

    from flashrank import RerankRequest
    passages = [
        {"id": doc_id, "text": doc_text}
        for doc_id, doc_text in zip(ids, documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    scored = [(r["id"], r["score"]) for r in results]
    return scored[:top_k] if top_k else scored
