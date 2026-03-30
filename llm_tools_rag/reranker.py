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

from llm_tools_rag.config import get_rag_cache_dir

# Lazy-loaded ranker instance, keyed by (model_name, max_length)
_ranker = None
_ranker_key = None


def _get_ranker(model_name: str, max_length: int = 512):
    """
    Get or create FlashRank Ranker instance (lazy-loaded).

    The model is downloaded on first use and cached locally.
    Recreated when model_name or max_length changes.
    """
    global _ranker, _ranker_key

    key = (model_name, max_length)
    if _ranker is None or _ranker_key != key:
        from flashrank import Ranker
        cache_dir = get_rag_cache_dir() / "flashrank"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _ranker = Ranker(model_name=model_name, cache_dir=str(cache_dir), max_length=max_length)
        _ranker_key = key

    return _ranker


def rerank_with_scores(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "ms-marco-MultiBERT-L-12",
    top_k: Optional[int] = None,
    max_length: int = 512
) -> List[tuple]:
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
        max_length: Max input length in tokens for the cross-encoder

    Returns:
        List of (doc_id, score) tuples sorted by score (highest first)

    Example:
        >>> scored = rerank_with_scores(
        ...     query="What is Python?",
        ...     documents=["Python is a snake", "Python is a programming language"],
        ...     ids=["doc1", "doc2"]
        ... )
        >>> scored[0][0]  # Most relevant document
        'doc2'
    """
    if not documents or len(documents) != len(ids):
        return [(doc_id, 0.0) for doc_id in (ids[:top_k] if top_k else ids)]

    ranker = _get_ranker(model_name, max_length=max_length)

    from flashrank import RerankRequest
    passages = [
        {"id": doc_id, "text": doc_text}
        for doc_id, doc_text in zip(ids, documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    scored = [(r["id"], r["score"]) for r in results]
    return scored[:top_k] if top_k else scored


def rerank(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "ms-marco-MultiBERT-L-12",
    top_k: Optional[int] = None,
    max_length: int = 512
) -> List[str]:
    """
    Rerank documents and return only the reordered IDs.

    Convenience wrapper around rerank_with_scores() that discards scores.

    Returns:
        List of document IDs reordered by relevance score (highest first)
    """
    return [
        doc_id for doc_id, _ in rerank_with_scores(
            query=query, documents=documents, ids=ids,
            model_name=model_name, top_k=top_k, max_length=max_length
        )
    ]
