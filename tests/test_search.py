"""
Tests for hybrid search functionality.
"""

import pytest
from llm_tools_rag.search import (
    tokenize, reciprocal_rank_fusion, BM25Index,
    _detect_script, _tokenize_cjk
)


def test_tokenize():
    """Test text tokenization for BM25."""
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # Punctuation should be excluded
    assert "," not in tokens


def test_tokenize_english_unchanged():
    """Existing English tokenization behavior should be preserved."""
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    # Stopwords removed ('the', 'over')
    assert "the" not in tokens
    # Stemming applied
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_explicit_language():
    """Explicit language parameter should override auto-detection."""
    tokens = tokenize("Der schnelle braune Fuchs", language="german")
    # German stopwords like 'der' should be removed
    assert "der" not in tokens
    assert len(tokens) > 0


def test_detect_script_latin():
    """Latin text should be detected as latin."""
    assert _detect_script("Hello world") == "latin"
    assert _detect_script("Bonjour le monde") == "latin"


def test_detect_script_cjk():
    """CJK text should be detected as CJK."""
    assert _detect_script("搜索引擎") == "cjk"
    assert _detect_script("こんにちは") == "cjk"  # Hiragana
    assert _detect_script("カタカナ") == "cjk"  # Katakana
    assert _detect_script("한국어") == "cjk"  # Hangul


def test_detect_script_mixed():
    """Mixed text should be classified by dominant script."""
    # More CJK than Latin
    assert _detect_script("搜索引擎优化 ok") == "cjk"
    # More Latin than CJK
    assert _detect_script("This is a test with one 字") == "latin"
    # Equal counts — falls through to latin (safe default)
    assert _detect_script("搜索引擎 test") == "latin"


def test_tokenize_cjk_bigrams():
    """CJK text should produce unigrams + overlapping bigrams."""
    tokens = _tokenize_cjk("搜索引擎")
    # Unigrams: 搜, 索, 引, 擎
    assert "搜" in tokens
    assert "索" in tokens
    assert "引" in tokens
    assert "擎" in tokens
    # Bigrams: 搜索, 索引, 引擎
    assert "搜索" in tokens
    assert "索引" in tokens
    assert "引擎" in tokens


def test_tokenize_cjk_mixed():
    """Mixed CJK+Latin text should tokenize both parts."""
    tokens = _tokenize_cjk("Python搜索引擎")
    # Latin part
    assert "python" in tokens
    # CJK unigrams
    assert "搜" in tokens
    # CJK bigrams
    assert "搜索" in tokens


def test_tokenize_cjk_autodetect():
    """Tokenize should auto-detect CJK and use bigram tokenization."""
    tokens = tokenize("搜索引擎")
    assert "搜索" in tokens
    assert "引擎" in tokens


def test_bm25_cjk_search():
    """BM25 index should work with CJK tokenized documents."""
    index = BM25Index(algorithm="plus")
    docs = [
        (0, "搜索引擎优化"),
        (1, "数据库管理系统"),
        (2, "Python programming language"),
    ]
    index.add_documents(docs)
    index.finalize()

    # Search for CJK content
    results = index.search("搜索引擎", top_k=2)
    assert 0 in results  # Document about search engines should match


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
    index = BM25Index(algorithm="plus")

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


def test_bm25_plus_positive_scores():
    """BM25Plus should return positive scores for matched terms.

    Unlike BM25Okapi which can return zero scores for terms appearing
    in many documents, BM25Plus guarantees positive scores.
    """
    # Create a corpus where a term appears in all documents
    index = BM25Index(algorithm="plus")
    docs = [
        (0, "python programming language"),
        (1, "python data science"),
        (2, "python web framework"),
    ]
    index.add_documents(docs)
    index.finalize()

    # "python" appears in all docs -- BM25Okapi would give zero scores,
    # but BM25Plus should still return results
    results = index.search("python", top_k=3)
    assert len(results) == 3  # All docs should match


def test_bm25_okapi_algorithm():
    """BM25Okapi algorithm should still work when explicitly selected."""
    index = BM25Index(algorithm="okapi")
    docs = [
        (0, "The quick brown fox jumps over the fence"),
        (1, "Slow green turtles swim in the ocean"),
        (2, "The lazy cat sleeps all day long"),
        (3, "Red birds fly south in winter time"),
    ]
    index.add_documents(docs)
    index.finalize()

    # "fox" appears only in doc 0, so BM25Okapi should score it positively
    results = index.search("fox", top_k=2)
    assert len(results) > 0
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
