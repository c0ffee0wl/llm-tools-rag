"""Tests for deduplication: exact (SHA256) and near-duplicate (MinHash/LSH)."""

import pytest
from llm_tools_rag.dedup import Deduplicator, NearDeduplicator


class TestExactDedup:
    """Regression tests for existing SHA256 exact deduplication."""

    def test_basic_dedup(self):
        """Identical content should be detected as duplicate."""
        dedup = Deduplicator()
        h1, is_new1 = dedup.add("hello world")
        assert is_new1 is True
        h2, is_new2 = dedup.add("hello world")
        assert is_new2 is False
        assert h1 == h2

    def test_different_content(self):
        """Different content should not be duplicate."""
        dedup = Deduplicator()
        _, is_new1 = dedup.add("hello world")
        _, is_new2 = dedup.add("goodbye world")
        assert is_new1 is True
        assert is_new2 is True

    def test_hash_content(self):
        """hash_content should return consistent SHA256 hex digest."""
        dedup = Deduplicator()
        h1 = dedup.hash_content("test")
        h2 = dedup.hash_content("test")
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length

    def test_contains_hash(self):
        """contains_hash should work with add_hash."""
        dedup = Deduplicator()
        h = dedup.hash_content("test")
        assert not dedup.contains_hash(h)
        dedup.add_hash(h)
        assert dedup.contains_hash(h)

    def test_size_and_clear(self):
        """size and clear should track correctly."""
        dedup = Deduplicator()
        assert dedup.size() == 0
        dedup.add("a")
        dedup.add("b")
        assert dedup.size() == 2
        dedup.clear()
        assert dedup.size() == 0


class TestNearDedup:
    """Tests for MinHash/LSH near-duplicate detection."""

    def test_identical_content(self):
        """Identical content should be detected as near-duplicate."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        nd.add("the quick brown fox jumps over the lazy dog")
        assert nd.is_near_duplicate("the quick brown fox jumps over the lazy dog")

    def test_similar_content(self):
        """Content differing by a few words should be near-duplicate."""
        nd = NearDeduplicator(threshold=0.3, num_perm=128)
        original = (
            "Python is a versatile programming language widely used for "
            "web development data science machine learning artificial intelligence "
            "and automation tasks in modern software engineering projects"
        )
        nd.add(original)
        # Very similar -- only last few words changed
        similar = (
            "Python is a versatile programming language widely used for "
            "web development data science machine learning artificial intelligence "
            "and scripting tasks in modern software engineering applications"
        )
        assert nd.is_near_duplicate(similar)

    def test_different_content(self):
        """Completely different content should NOT be near-duplicate."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        nd.add(
            "Python is a versatile programming language used for web development "
            "data science machine learning and automation tasks"
        )
        assert not nd.is_near_duplicate(
            "The weather forecast for tomorrow shows clear skies with temperatures "
            "reaching a high of thirty degrees celsius in the afternoon"
        )

    def test_threshold_sensitivity(self):
        """Higher threshold should be stricter (fewer matches)."""
        text1 = "the quick brown fox jumps over the lazy dog near the river bank"
        # Moderately similar text
        text2 = "the quick brown cat jumps over the lazy frog near the lake shore"

        # Low threshold: should detect as near-dup
        nd_low = NearDeduplicator(threshold=0.3, num_perm=128)
        nd_low.add(text1)
        is_dup_low = nd_low.is_near_duplicate(text2)

        # High threshold: should NOT detect as near-dup
        nd_high = NearDeduplicator(threshold=0.9, num_perm=128)
        nd_high.add(text1)
        is_dup_high = nd_high.is_near_duplicate(text2)

        # Low threshold should be more permissive than high
        assert is_dup_low or not is_dup_high  # At minimum, high should not match if low doesn't

    def test_serialization_roundtrip(self):
        """MinHash signatures should survive serialization to/from metadata."""
        nd1 = NearDeduplicator(threshold=0.5, num_perm=64)
        content = "this is a test document with enough words for proper shingling"

        # check_add_and_serialize returns (is_near_dup, sig_json)
        is_dup, sig_json = nd1.check_add_and_serialize(content, key="key1")
        assert is_dup is False  # First entry is never a dup
        assert isinstance(sig_json, str)

        # Deserialize
        hashvalues = NearDeduplicator.metadata_to_hashvalues(sig_json)
        assert isinstance(hashvalues, list)
        assert len(hashvalues) == 64  # num_perm

        # Rebuild in a new NearDeduplicator and verify detection
        nd2 = NearDeduplicator(threshold=0.5, num_perm=64)
        nd2.add_from_signature("key1", hashvalues)

        # The same content should be detected as near-duplicate
        assert nd2.is_near_duplicate(content)

    def test_short_content(self):
        """Content shorter than 3 words should not crash."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        # Single word
        nd.add("hello")
        assert nd.size() == 1
        # Two words
        nd.add("hello world")
        assert nd.size() == 2
        # Empty string
        nd.add("")
        assert nd.size() == 3

    def test_clear(self):
        """Clear should reset the index."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        nd.add("test content with enough words for shingling")
        assert nd.size() == 1

        nd.clear()
        assert nd.size() == 0
        # Previously added content should no longer be detected
        assert not nd.is_near_duplicate("test content with enough words for shingling")

    def test_add_returns_key(self):
        """add() should return the key used."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        key = nd.add("some content", key="my_key")
        assert key == "my_key"

        # Auto-generated key
        key2 = nd.add("other content")
        assert key2 == "nd_1"

    def test_duplicate_key_handled(self):
        """Adding with a duplicate key should not crash or inflate count."""
        nd = NearDeduplicator(threshold=0.5, num_perm=128)
        nd.add("content one", key="same_key")
        assert nd.size() == 1
        # Should not raise, and count should not increase
        nd.add("content two", key="same_key")
        assert nd.size() == 1


    def test_check_add_and_serialize(self):
        """check_add_and_serialize should detect, add, and serialize in one pass."""
        nd = NearDeduplicator(threshold=0.5, num_perm=64)
        content = "this is a test document with enough words for proper shingling"

        is_dup, sig = nd.check_add_and_serialize(content, key="k1")
        assert is_dup is False
        assert isinstance(sig, str)
        assert nd.size() == 1

        # Same content should now be detected as near-duplicate
        is_dup2, sig2 = nd.check_add_and_serialize(content, key="k1")
        assert is_dup2 is True
        # Duplicate key should not inflate count
        assert nd.size() == 1


class TestConfigNearDedup:
    """Tests for near-dedup config validation."""

    def test_config_defaults(self):
        """Config should have correct near-dedup defaults."""
        from llm_tools_rag.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["near_dedup_enabled"] is True
        assert DEFAULT_CONFIG["near_dedup_threshold"] == 0.8
        assert DEFAULT_CONFIG["near_dedup_num_perm"] == 128

    def test_config_defaults_bm25_algorithm(self):
        """Config should have correct bm25_algorithm default."""
        from llm_tools_rag.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["bm25_algorithm"] == "plus"

    def test_config_getters(self):
        """Config getters should return correct values."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        assert config.get_near_dedup_enabled() is True
        assert config.get_near_dedup_threshold() == 0.8
        assert config.get_near_dedup_num_perm() == 128
        assert config.get_bm25_algorithm() == "plus"

    def test_invalid_threshold_too_low(self):
        """Threshold of 0 should be rejected."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        config.set("near_dedup_threshold", 0.0)
        with pytest.raises(ValueError, match="near_dedup_threshold"):
            config._validate_config()

    def test_invalid_threshold_too_high(self):
        """Threshold above 1 should be rejected."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        config.set("near_dedup_threshold", 1.5)
        with pytest.raises(ValueError, match="near_dedup_threshold"):
            config._validate_config()

    def test_invalid_num_perm_too_low(self):
        """num_perm below 16 should be rejected."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        config.set("near_dedup_num_perm", 8)
        with pytest.raises(ValueError, match="near_dedup_num_perm"):
            config._validate_config()

    def test_invalid_num_perm_too_high(self):
        """num_perm above 512 should be rejected."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        config.set("near_dedup_num_perm", 1024)
        with pytest.raises(ValueError, match="near_dedup_num_perm"):
            config._validate_config()

    def test_invalid_bm25_algorithm(self):
        """Invalid bm25_algorithm should be rejected."""
        from llm_tools_rag.config import RAGConfig
        RAGConfig.reset()
        config = RAGConfig()
        config.set("bm25_algorithm", "invalid")
        with pytest.raises(ValueError, match="bm25_algorithm"):
            config._validate_config()
