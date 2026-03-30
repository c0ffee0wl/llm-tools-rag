"""
SHA256-based deduplication for documents and chunks.
Matches aichat's approach to avoid reindexing identical content.

Also provides MinHash-based near-duplicate detection via datasketch
for catching overlapping chunks and reformatted content.
"""

import hashlib
import json
from typing import Set, List, Tuple, Optional


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


class NearDeduplicator:
    """MinHash/LSH-based near-duplicate detector for chunks.

    Uses word-level 3-gram shingles and MinHashLSH from datasketch
    to detect content that is nearly identical (e.g., overlapping chunks
    from different sources, or reformatted versions of the same text).

    Complementary to the exact-match Deduplicator -- use both together.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        from datasketch import MinHashLSH
        self._MinHash = None  # lazy-loaded reference to MinHash class
        self._np = None  # lazy-loaded reference to numpy
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._count = 0

    def _get_minhash_cls(self):
        if self._MinHash is None:
            from datasketch import MinHash
            self._MinHash = MinHash
        return self._MinHash

    def _get_numpy(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    def _create_minhash(self, content: str):
        """Create a MinHash signature from text using word-level 3-gram shingles."""
        MinHash = self._get_minhash_cls()
        m = MinHash(num_perm=self.num_perm)
        words = content.lower().split()
        if len(words) < 3:
            for w in words:
                m.update(w.encode('utf-8'))
        else:
            for i in range(len(words) - 2):
                shingle = " ".join(words[i:i + 3])
                m.update(shingle.encode('utf-8'))
        return m

    def _insert(self, key: str, minhash) -> bool:
        """Insert minhash into LSH. Returns True if inserted, False if key exists."""
        try:
            self.lsh.insert(key, minhash)
            self._count += 1
            return True
        except ValueError:
            return False  # Key already exists in LSH

    def is_near_duplicate(self, content: str) -> bool:
        """Check if content is a near-duplicate of any indexed content."""
        minhash = self._create_minhash(content)
        return len(self.lsh.query(minhash)) > 0

    def add(self, content: str, key: Optional[str] = None) -> str:
        """Add content to the near-dedup index.

        Returns:
            The key used for this entry
        """
        if key is None:
            key = f"nd_{self._count}"
        minhash = self._create_minhash(content)
        self._insert(key, minhash)
        return key

    def check_add_and_serialize(self, content: str, key: str) -> tuple:
        """Check near-dup, add to index, and serialize signature in one pass.

        Computes the MinHash once and reuses it for all three operations,
        avoiding the 3x cost of calling is_near_duplicate + add +
        create_minhash_metadata separately.

        Returns:
            (is_near_dup: bool, minhash_sig_json: str)
        """
        minhash = self._create_minhash(content)
        is_near_dup = len(self.lsh.query(minhash)) > 0
        self._insert(key, minhash)
        sig_json = json.dumps(minhash.hashvalues.tolist())
        return is_near_dup, sig_json

    def add_from_signature(self, key: str, hashvalues: List[int]):
        """Rebuild a MinHash entry from stored signature values.

        Used to reload the LSH index from ChromaDB metadata on startup.
        """
        MinHash = self._get_minhash_cls()
        np = self._get_numpy()
        m = MinHash(num_perm=self.num_perm)
        m.hashvalues = np.array(hashvalues, dtype=np.uint64)
        self._insert(key, m)

    @staticmethod
    def metadata_to_hashvalues(metadata_str: str) -> List[int]:
        """Deserialize MinHash hashvalues from a metadata JSON string."""
        return json.loads(metadata_str)

    def clear(self):
        """Clear all tracked signatures."""
        from datasketch import MinHashLSH
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._count = 0

    def size(self) -> int:
        """Get number of signatures tracked."""
        return self._count
