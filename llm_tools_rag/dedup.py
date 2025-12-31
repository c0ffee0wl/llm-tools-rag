"""
SHA256-based deduplication for documents and chunks.
Matches aichat's approach to avoid reindexing identical content.
"""

import hashlib
from typing import Set, List, Tuple


class Deduplicator:
    """SHA256-based content deduplicator."""

    def __init__(self):
        """Initialize deduplicator with empty hash registry."""
        self.seen_hashes: Set[str] = set()
        # Note: hash_map removed to prevent memory leak
        # Previously stored full content (gigabytes for large collections)
        # Only hash tracking is needed for deduplication

    def hash_content(self, content: str) -> str:
        """
        Compute SHA256 hash of content.

        Args:
            content: Text content to hash

        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add(self, content: str) -> Tuple[str, bool]:
        """
        Add content and check if it's a duplicate.

        Args:
            content: Text content to check

        Returns:
            Tuple of (hash, is_new) where is_new is True if content is novel
        """
        content_hash = self.hash_content(content)

        if content_hash in self.seen_hashes:
            return content_hash, False

        self.seen_hashes.add(content_hash)
        return content_hash, True

    def filter_duplicates(self, contents: List[str]) -> List[Tuple[str, str]]:
        """
        Filter out duplicate content from a list.

        Args:
            contents: List of text content

        Returns:
            List of (hash, content) tuples for unique content only
        """
        unique = []
        for content in contents:
            content_hash, is_new = self.add(content)
            if is_new:
                unique.append((content_hash, content))
        return unique

    def is_duplicate(self, content: str) -> bool:
        """
        Check if content is a duplicate without adding it.

        Args:
            content: Text content to check

        Returns:
            True if content has been seen before
        """
        content_hash = self.hash_content(content)
        return content_hash in self.seen_hashes

    def contains_hash(self, content_hash: str) -> bool:
        """Check if a specific hash is in the registry."""
        return content_hash in self.seen_hashes

    def add_hash(self, content_hash: str):
        """Add a hash to the registry (when loading existing data)."""
        self.seen_hashes.add(content_hash)

    def clear(self):
        """Clear all tracked hashes."""
        self.seen_hashes.clear()

    def size(self) -> int:
        """Get number of unique content hashes tracked."""
        return len(self.seen_hashes)
