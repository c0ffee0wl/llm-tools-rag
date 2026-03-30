"""
Hybrid search with Reciprocal Rank Fusion (RRF).
Combines vector similarity (ChromaDB) with keyword matching (BM25).
Algorithm ported from aichat's implementation.
"""

from typing import List, Dict, Tuple, Any, Optional
from rank_bm25 import BM25Okapi
import re

from llm_tools_rag.config import DEFAULT_CONFIG

# Lazy-loaded stemmers and stopwords per language (avoid import overhead until first use)
_stemmers: Dict[str, Any] = {}
_stopwords_cache: Dict[str, set] = {}


def _get_stemmer(language: str = "english"):
    """Get or create SnowballStemmer instance for the given language (lazy-loaded)."""
    if language not in _stemmers:
        from nltk.stem import SnowballStemmer
        try:
            _stemmers[language] = SnowballStemmer(language)
        except ValueError:
            # Unsupported language — fall back to English
            _stemmers[language] = _get_stemmer("english")
    return _stemmers[language]


def _get_stopwords(language: str = "english"):
    """Get stopwords set for the given language (lazy-loaded with automatic download)."""
    if language not in _stopwords_cache:
        try:
            from nltk.corpus import stopwords
            _stopwords_cache[language] = set(stopwords.words(language))
        except LookupError:
            import nltk
            nltk.download('stopwords', quiet=True)
            try:
                from nltk.corpus import stopwords
                _stopwords_cache[language] = set(stopwords.words(language))
            except Exception:
                _stopwords_cache[language] = set()
        except Exception:
            # Language not available in NLTK stopwords — use empty set
            # (CJK benefits from natural IDF filtering anyway)
            _stopwords_cache[language] = set()
    return _stopwords_cache[language]


def _is_cjk_char(cp: int) -> bool:
    """Check if a Unicode codepoint is a CJK character."""
    return (
        (0x4E00 <= cp <= 0x9FFF)     # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0xF900 <= cp <= 0xFAFF)  # CJK Compatibility Ideographs
        or (0x3040 <= cp <= 0x309F)  # Hiragana
        or (0x30A0 <= cp <= 0x30FF)  # Katakana
        or (0xAC00 <= cp <= 0xD7A3)  # Hangul Syllables
    )


def _detect_script(text: str) -> str:
    """Detect dominant script in text. Returns 'cjk' or 'latin'.

    Samples only the first 200 characters to avoid scanning large texts.
    """
    cjk_count = 0
    latin_count = 0
    for char in text[:200]:
        cp = ord(char)
        if _is_cjk_char(cp):
            cjk_count += 1
        elif (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A
              or 0x00C0 <= cp <= 0x024F):
            latin_count += 1
    if cjk_count > latin_count and cjk_count > 0:
        return "cjk"
    return "latin"


def _tokenize_cjk(text: str) -> List[str]:
    """Tokenize text containing CJK characters using Lucene-style bigrams.

    CJK segments get overlapping character unigrams + bigrams.
    Latin segments get standard word splitting + lowercasing.
    """
    tokens = []
    cjk_chars: List[str] = []
    latin_buf: List[str] = []

    def flush_cjk():
        # Emit unigrams + overlapping bigrams (Lucene CJKBigramFilter approach)
        for i in range(len(cjk_chars)):
            tokens.append(cjk_chars[i])
            if i < len(cjk_chars) - 1:
                tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        cjk_chars.clear()

    def flush_latin():
        if latin_buf:
            word = ''.join(latin_buf).lower()
            if len(word) > 1:
                tokens.append(word)
            latin_buf.clear()

    for char in text:
        cp = ord(char)
        if _is_cjk_char(cp):
            flush_latin()
            cjk_chars.append(char)
        elif char.isalnum():
            if cjk_chars:
                flush_cjk()
            latin_buf.append(char)
        else:
            if cjk_chars:
                flush_cjk()
            flush_latin()

    # Flush remaining
    flush_cjk()
    flush_latin()
    return tokens


def tokenize(text: str, language: Optional[str] = None) -> List[str]:
    """
    Tokenize text for BM25 with stemming and stopword removal.

    Supports multilingual text:
    - CJK (Chinese, Japanese, Korean): character unigrams + overlapping bigrams
    - Latin-script languages: SnowballStemmer (16 languages) + NLTK stopwords
    - Auto-detects script when language is None
    """
    if language is None:
        script = _detect_script(text)
        if script == "cjk":
            return _tokenize_cjk(text)
        language = "english"

    # Latin-script tokenization with stemming and stopword removal
    tokens = re.findall(r'[\w]+', text.lower(), re.UNICODE)
    stemmer = _get_stemmer(language)
    stops = _get_stopwords(language)
    return [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]


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
            List of document IDs ranked by BM25 score (excludes zero-score documents)
        """
        if not self.bm25 or not self.corpus_tokens:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices sorted by score (descending), filtering out zero scores
        # Documents with score=0 have no keyword relevance and shouldn't be returned
        top_indices = sorted(
            (i for i in range(len(scores)) if scores[i] > 0),
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
        top_k: int,
        candidate_count: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None
    ) -> List[int]:
        """
        Perform hybrid search combining vector and keyword results.

        Args:
            query: Search query
            vector_results: Pre-computed vector search results (list of doc IDs)
            top_k: Number of final results to return
            candidate_count: Number of BM25 candidates to retrieve (None = top_k * 2)
            vector_weight: Override vector weight for this search (None = use default)
            keyword_weight: Override keyword weight for this search (None = use default)

        Returns:
            Fused list of document IDs
        """
        bm25_results = self.bm25_index.search(query, top_k=candidate_count or top_k * 2)

        if not bm25_results:
            return vector_results[:top_k]

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            weights=[
                vector_weight if vector_weight is not None else self.vector_weight,
                keyword_weight if keyword_weight is not None else self.keyword_weight,
            ],
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
