"""
Hybrid search with Reciprocal Rank Fusion (RRF).
Combines vector similarity (ChromaDB) with keyword matching (BM25).
Algorithm ported from aichat's implementation.
"""

from typing import List, Dict, Tuple, Any, Optional
from rank_bm25 import BM25Okapi
import re

from llm_tools_rag.config import DEFAULT_CONFIG


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 with Unicode support.
    Supports international text including Chinese, Japanese, Arabic, etc.
    """
    # Lowercase and split on word characters with Unicode support
    # re.UNICODE flag makes \w match Unicode word characters
    tokens = re.findall(r'[\w]+', text.lower(), re.UNICODE)
    return tokens


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    weights: List[float],
    rrf_k: int,
    top_k: int
) -> List[int]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Ported from aichat's implementation:
    https://github.com/sigoden/aichat/blob/main/src/rag/mod.rs#L990

    Args:
        ranked_lists: List of ranked document ID lists
        weights: Weight for each ranked list
        rrf_k: RRF constant (should be constant for consistent scoring)
        top_k: Number of results to return

    Returns:
        Fused ranked list of document IDs

    Raises:
        TypeError: If ranked_lists or weights have incorrect types
        ValueError: If lengths don't match or values are invalid
    """
    # Validate input types
    if not isinstance(ranked_lists, (list, tuple)):
        raise TypeError(f"ranked_lists must be a list, got {type(ranked_lists).__name__}")

    if not isinstance(weights, (list, tuple)):
        raise TypeError(f"weights must be a list, got {type(weights).__name__}")

    # Validate lengths match
    if len(ranked_lists) != len(weights):
        raise ValueError(
            f"ranked_lists ({len(ranked_lists)}) and weights ({len(weights)}) "
            f"must have the same length"
        )

    # Validate ranked_lists structure (list of lists)
    for i, ranked_list in enumerate(ranked_lists):
        if not isinstance(ranked_list, (list, tuple)):
            raise TypeError(
                f"ranked_lists[{i}] must be a list or tuple, "
                f"got {type(ranked_list).__name__}"
            )

    # Validate weights are valid numbers
    for i, weight in enumerate(weights):
        if not isinstance(weight, (int, float)):
            raise TypeError(
                f"weights[{i}] must be a number, got {type(weight).__name__}"
            )
        if weight < 0:
            raise ValueError(
                f"weights[{i}] must be non-negative, got {weight}"
            )

    # Normalize weights to sum to 1.0 for consistent scoring behavior
    # This ensures weights represent relative importance, not absolute multipliers
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    # Build score map using RRF formula: score = sum(weight / (rrf_k + rank + 1))
    # Use the provided rrf_k constant for consistent scoring across different top_k values
    score_map: Dict[int, float] = {}

    for doc_ids, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(doc_ids):
            score = weight / (rrf_k + rank + 1)
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score

    # Sort by score (descending) and return top_k
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_items[:top_k]]


class BM25Index:
    """BM25 keyword search index."""

    def __init__(self):
        """Initialize empty BM25 index."""
        self.corpus_tokens: List[List[str]] = []
        self.document_ids: List[int] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_documents(self, documents: List[Tuple[int, str]]):
        """
        Add documents to the BM25 index.
        Note: Call finalize() after all documents are added to build the index.

        Args:
            documents: List of (doc_id, text) tuples
        """
        for doc_id, text in documents:
            tokens = tokenize(text)
            self.corpus_tokens.append(tokens)
            self.document_ids.append(doc_id)

    def finalize(self):
        """
        Finalize the BM25 index after all documents have been added.
        This should be called once after batch adding documents to avoid
        rebuilding the index multiple times (memory leak).
        """
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[int]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of document IDs ranked by BM25 score
        """
        if not self.bm25 or not self.corpus_tokens:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top_k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Map indices to document IDs
        return [self.document_ids[i] for i in top_indices]

    def clear(self):
        """Clear the index."""
        self.corpus_tokens = []
        self.document_ids = []
        self.bm25 = None

    def size(self) -> int:
        """Get number of documents in index."""
        return len(self.document_ids)


class HybridSearch:
    """
    Hybrid search combining vector similarity and keyword matching.
    """

    def __init__(
        self,
        vector_weight: float,
        keyword_weight: float,
        rrf_k: int
    ):
        """
        Initialize hybrid search.

        Args:
            vector_weight: Weight for vector search results
            keyword_weight: Weight for BM25 keyword search results
            rrf_k: RRF constant for rank fusion
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.bm25_index = BM25Index()

    def search(
        self,
        query: str,
        vector_results: List[int],
        top_k: int
    ) -> List[int]:
        """
        Perform hybrid search combining vector and keyword results.

        Args:
            query: Search query
            vector_results: Pre-computed vector search results (list of doc IDs)
            top_k: Number of final results to return

        Returns:
            Fused list of document IDs
        """
        # Get BM25 results
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # If no BM25 index or results, return vector results only
        if not bm25_results:
            return vector_results[:top_k]

        # Combine using RRF
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            weights=[self.vector_weight, self.keyword_weight],
            rrf_k=self.rrf_k,
            top_k=top_k
        )

        return fused_results

    def add_documents(self, documents: List[Tuple[int, str]]):
        """Add documents to BM25 index. Call finalize() after all adds."""
        self.bm25_index.add_documents(documents)

    def finalize(self):
        """Finalize the BM25 index after all documents have been added."""
        self.bm25_index.finalize()

    def clear(self):
        """Clear BM25 index."""
        self.bm25_index.clear()


def create_hybrid_searcher(config: Dict[str, Any]) -> HybridSearch:
    """
    Create hybrid searcher from configuration.

    Args:
        config: Configuration dictionary with optional keys:
            - vector_weight: Weight for vector search
            - keyword_weight: Weight for keyword search
            - rrf_k: RRF constant

    Returns:
        Configured HybridSearch instance
    """
    return HybridSearch(
        vector_weight=config.get('vector_weight', DEFAULT_CONFIG['vector_weight']),
        keyword_weight=config.get('keyword_weight', DEFAULT_CONFIG['keyword_weight']),
        rrf_k=config.get('rrf_k', DEFAULT_CONFIG['rrf_k'])
    )
