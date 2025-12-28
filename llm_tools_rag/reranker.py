"""
Cross-encoder reranking for improved relevance.

Cross-encoders score (query, document) pairs directly, providing more accurate
relevance judgments than embedding similarity alone. They are embedding-model
agnostic - any cross-encoder works with any embedding model.

Default model: BAAI/bge-reranker-v2-m3
- Multilingual: 100+ languages (handles German, Chinese, etc.)
- High quality: MIRACL score 69.32
- Size: ~560MB download on first use
"""

from typing import List, Optional

# Lazy-loaded cross-encoder instance
_cross_encoder = None
_cross_encoder_model_name = None


def _get_cross_encoder(model_name: str):
    """
    Get or create CrossEncoder instance (lazy-loaded).

    The model is loaded on first use and cached for subsequent calls.
    Model files are downloaded automatically from HuggingFace on first use.

    Args:
        model_name: HuggingFace model ID (e.g., "BAAI/bge-reranker-v2-m3")

    Returns:
        CrossEncoder instance
    """
    global _cross_encoder, _cross_encoder_model_name

    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name

    return _cross_encoder


def rerank(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: Optional[int] = None
) -> List[str]:
    """
    Rerank documents using cross-encoder relevance scores.

    Cross-encoders process (query, document) pairs together, allowing them to
    capture fine-grained relevance signals that bi-encoder embeddings miss.

    Args:
        query: The search query
        documents: List of document texts to rerank
        ids: List of document IDs corresponding to documents
        model_name: HuggingFace cross-encoder model ID
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

    # Get cross-encoder (lazy-loaded)
    cross_encoder = _get_cross_encoder(model_name)

    # Create (query, document) pairs for scoring
    pairs = [(query, doc) for doc in documents]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)

    # Sort by score (descending) and extract IDs
    scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    result = [doc_id for doc_id, _ in scored]

    # Apply top_k limit if specified
    return result[:top_k] if top_k else result


def rerank_with_scores(
    query: str,
    documents: List[str],
    ids: List[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: Optional[int] = None
) -> List[tuple]:
    """
    Rerank documents and return scores along with IDs.

    Same as rerank() but returns (id, score) tuples for debugging/analysis.

    Args:
        query: The search query
        documents: List of document texts to rerank
        ids: List of document IDs corresponding to documents
        model_name: HuggingFace cross-encoder model ID
        top_k: Number of top results to return (None = return all)

    Returns:
        List of (doc_id, score) tuples sorted by score (highest first)
    """
    if not documents or len(documents) != len(ids):
        return [(doc_id, 0.0) for doc_id in (ids[:top_k] if top_k else ids)]

    cross_encoder = _get_cross_encoder(model_name)
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k] if top_k else scored
