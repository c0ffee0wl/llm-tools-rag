"""
Query analysis for dynamic weight adjustment.

Detects query types and adjusts vector/keyword weights accordingly:
- Entity/mention queries (e.g., "Does X mention Y?") → boost keyword search
- Semantic queries (e.g., "How does X work?") → boost vector search
"""

from typing import Tuple
import re


# Patterns indicating entity/mention queries (boost keyword search)
# These queries look for specific names, terms, or references
ENTITY_PATTERNS = [
    r'\b(mention|erwähnt|talk about|reference|who is|wer ist)\b',
    r'\b(named?|called|heißt|genannt)\b',
    r'\b(refers? to|bezieht sich auf)\b',
    r'\b(contains?|enthält)\b',
    r'\b(says?|sagt|said)\b',
]

# Patterns indicating semantic/conceptual queries (boost vector search)
# These queries seek understanding, explanation, or related concepts
SEMANTIC_PATTERNS = [
    r'\b(how|why|explain|what is|describe|wie|warum|erkläre)\b',
    r'\b(similar|like|related|ähnlich|verwandt)\b',
    r'\b(concept|idea|approach|konzept|ansatz)\b',
    r'\b(difference|unterschied|compare|vergleich)\b',
    r'\b(overview|summary|zusammenfassung|überblick)\b',
]


def analyze_query(query: str) -> Tuple[float, float]:
    """
    Analyze query and return appropriate search weights.

    Args:
        query: The search query string

    Returns:
        Tuple of (vector_weight, keyword_weight) that sum to 1.0
        - Entity queries: (0.3, 0.7) - keyword-heavy
        - Semantic queries: (0.8, 0.2) - vector-heavy
        - Default: (0.5, 0.5) - balanced
    """
    query_lower = query.lower()

    # Check for entity/mention patterns → boost keyword search
    for pattern in ENTITY_PATTERNS:
        if re.search(pattern, query_lower):
            return (0.3, 0.7)  # Keyword-heavy for entity lookups

    # Check for semantic patterns → boost vector search
    for pattern in SEMANTIC_PATTERNS:
        if re.search(pattern, query_lower):
            return (0.8, 0.2)  # Vector-heavy for conceptual queries

    # Default: balanced hybrid search
    return (0.5, 0.5)


def get_query_type(query: str) -> str:
    """
    Get a descriptive label for the query type.

    Useful for debugging and logging.

    Args:
        query: The search query string

    Returns:
        One of: "entity", "semantic", or "balanced"
    """
    vector_weight, keyword_weight = analyze_query(query)

    if keyword_weight > vector_weight:
        return "entity"
    elif vector_weight > keyword_weight:
        return "semantic"
    else:
        return "balanced"
