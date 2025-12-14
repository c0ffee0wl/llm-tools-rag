"""
Tests for hybrid search functionality.
"""

import pytest
from llm_tools_rag.search import tokenize, reciprocal_rank_fusion, BM25Index


def test_tokenize():
    """Test text tokenization for BM25."""
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # Punctuation should be excluded
    assert "," not in tokens


def test_rrf_basic():
    """Test basic RRF fusion."""
    # Two ranking lists
    list1 = [1, 2, 3, 4, 5]
    list2 = [3, 1, 5, 2, 4]

    # Equal weights
    result = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        weights=[1.0, 1.0],
        rrf_k=60,
        top_k=3
    )

    # Should have 3 results
    assert len(result) == 3
    # Item 1 and 3 appear early in both lists
    assert 1 in result
    assert 3 in result


def test_rrf_weighted():
    """Test RRF with different weights."""
    list1 = [1, 2, 3]
    list2 = [3, 2, 1]

    # Heavily weight first list
    result = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        weights=[0.9, 0.1],
        rrf_k=60,
        top_k=3
    )

    # Item 1 should be ranked highly due to weight
    assert result[0] == 1


def test_bm25_index():
    """Test BM25 index functionality."""
    index = BM25Index()

    # Add documents
    docs = [
        (0, "The quick brown fox"),
        (1, "Quick brown dogs"),
        (2, "The lazy cat sleeps")
    ]
    index.add_documents(docs)
    # Finalize to build the BM25 index
    index.finalize()

    assert index.size() == 3

    # Search
    results = index.search("quick fox", top_k=2)
    assert len(results) <= 2
    # Document 0 should rank highly (has both words)
    assert 0 in results


def test_bm25_empty():
    """Test BM25 with empty index."""
    index = BM25Index()
    results = index.search("test", top_k=5)
    assert results == []


def test_bm25_clear():
    """Test clearing BM25 index."""
    index = BM25Index()
    index.add_documents([(0, "test document")])
    # Finalize to build the BM25 index
    index.finalize()
    assert index.size() == 1

    index.clear()
    assert index.size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
