"""
Maximal Marginal Relevance (MMR) for result diversification.

MMR balances relevance and diversity by iteratively selecting documents
that are both relevant to the query and different from already selected results.
"""

from typing import List
import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm_product < 1e-10:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm_product)


def maximal_marginal_relevance(
    query_embedding: List[float],
    embeddings: List[List[float]],
    ids: List[str],
    lambda_mult: float = 0.5,
    k: int = 5
) -> List[str]:
    """
    Select diverse results using Maximal Marginal Relevance.

    MMR iteratively selects documents that maximize:
        MMR = lambda * similarity(doc, query) - (1 - lambda) * max(similarity(doc, selected))

    Args:
        query_embedding: Query vector
        embeddings: Document embeddings (same order as ids)
        ids: Document IDs
        lambda_mult: Trade-off between relevance and diversity
                    - 1.0 = max relevance (no diversification)
                    - 0.0 = max diversity (ignore relevance)
                    - 0.5 = balanced (default)
        k: Number of results to return

    Returns:
        Reordered IDs balancing relevance and diversity
    """
    if not embeddings or not ids:
        return ids[:k]

    if len(embeddings) != len(ids):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and ids ({len(ids)}) must have same length"
        )

    # Validate all embeddings have the same dimension
    if embeddings:
        expected_dim = len(embeddings[0])
        for i, emb in enumerate(embeddings):
            if len(emb) != expected_dim:
                raise ValueError(
                    f"Embedding {i} has {len(emb)} dimensions, expected {expected_dim}"
                )

    # Compute relevance scores (query similarity)
    relevance = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    selected: List[int] = []
    remaining = list(range(len(ids)))

    for _ in range(min(k, len(ids))):
        best_score = -float('inf')
        best_idx = -1

        for idx in remaining:
            # Relevance component
            rel_score = relevance[idx]

            # Diversity component (max similarity to already selected)
            if selected:
                max_sim = max(
                    cosine_similarity(embeddings[idx], embeddings[s])
                    for s in selected
                )
            else:
                max_sim = 0.0

            # MMR score
            mmr_score = lambda_mult * rel_score - (1 - lambda_mult) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [ids[i] for i in selected]
