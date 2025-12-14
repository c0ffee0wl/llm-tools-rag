"""
Text chunking with language-aware recursive character splitting.
Ported from aichat's RecursiveCharacterTextSplitter implementation.
"""

from typing import List, Dict, Callable, Optional
from pathlib import Path


# Default separators for generic text
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


class Language:
    """Language-specific separators for code splitting."""

    @staticmethod
    def get_separators(language: str) -> List[str]:
        """Get separators for a specific language."""
        separators_map = {
            "cpp": ["\nclass ", "\nvoid ", "\nint ", "\nfloat ", "\ndouble ",
                   "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                   "\n\n", "\n", " ", ""],
            "c": ["\nclass ", "\nvoid ", "\nint ", "\nfloat ", "\ndouble ",
                 "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                 "\n\n", "\n", " ", ""],
            "cc": ["\nclass ", "\nvoid ", "\nint ", "\nfloat ", "\ndouble ",
                  "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                  "\n\n", "\n", " ", ""],
            "go": ["\nfunc ", "\nvar ", "\nconst ", "\ntype ", "\nif ",
                  "\nfor ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""],
            "java": ["\nclass ", "\npublic ", "\nprotected ", "\nprivate ",
                    "\nstatic ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ",
                    "\ncase ", "\n\n", "\n", " ", ""],
            "js": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ",
                  "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                  "\ndefault ", "\n\n", "\n", " ", ""],
            "mjs": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ",
                   "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                   "\ndefault ", "\n\n", "\n", " ", ""],
            "cjs": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ",
                   "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
                   "\ndefault ", "\n\n", "\n", " ", ""],
            "php": ["\nfunction ", "\nclass ", "\nif ", "\nforeach ",
                   "\nwhile ", "\ndo ", "\nswitch ", "\ncase ",
                   "\n\n", "\n", " ", ""],
            "proto": ["\nmessage ", "\nservice ", "\nenum ", "\noption ",
                     "\nimport ", "\nsyntax ", "\n\n", "\n", " ", ""],
            "py": ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""],
            "rst": ["\n===\n", "\n---\n", "\n***\n", "\n.. ", "\n\n", "\n", " ", ""],
            "rb": ["\ndef ", "\nclass ", "\nif ", "\nunless ", "\nwhile ",
                  "\nfor ", "\ndo ", "\nbegin ", "\nrescue ",
                  "\n\n", "\n", " ", ""],
            "rs": ["\nfn ", "\nconst ", "\nlet ", "\nif ", "\nwhile ",
                  "\nfor ", "\nloop ", "\nmatch ", "\nconst ",
                  "\n\n", "\n", " ", ""],
            "scala": ["\nclass ", "\nobject ", "\ndef ", "\nval ", "\nvar ",
                     "\nif ", "\nfor ", "\nwhile ", "\nmatch ", "\ncase ",
                     "\n\n", "\n", " ", ""],
            "swift": ["\nfunc ", "\nclass ", "\nstruct ", "\nenum ", "\nif ",
                     "\nfor ", "\nwhile ", "\ndo ", "\nswitch ", "\ncase ",
                     "\n\n", "\n", " ", ""],
            "md": ["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
                  "```\n\n", "\n\n***\n\n", "\n\n---\n\n", "\n\n___\n\n",
                  "\n\n", "\n", " ", ""],
            "mkd": ["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
                   "```\n\n", "\n\n***\n\n", "\n\n---\n\n", "\n\n___\n\n",
                   "\n\n", "\n", " ", ""],
            "tex": ["\n\\chapter{", "\n\\section{", "\n\\subsection{", "\n\\subsubsection{",
                   "\n\\begin{enumerate}", "\n\\begin{itemize}", "\n\\begin{description}",
                   "\n\\begin{list}", "\n\\begin{quote}", "\n\\begin{quotation}",
                   "\n\\begin{verse}", "\n\\begin{verbatim}", "\n\\begin{align}",
                   "$$", "$", "\n\n", "\n", " ", ""],
            "html": ["<body>", "<div>", "<p>", "<br>", "<li>", "<h1>", "<h2>", "<h3>",
                    "<h4>", "<h5>", "<h6>", "<span>", "<table>", "<tr>", "<td>", "<th>",
                    "<ul>", "<ol>", "<header>", "<footer>", "<nav>", "<head>", "<style>",
                    "<script>", "<meta>", "<title>", " ", ""],
            "htm": ["<body>", "<div>", "<p>", "<br>", "<li>", "<h1>", "<h2>", "<h3>",
                   "<h4>", "<h5>", "<h6>", "<span>", "<table>", "<tr>", "<td>", "<th>",
                   "<ul>", "<ol>", "<header>", "<footer>", "<nav>", "<head>", "<style>",
                   "<script>", "<meta>", "<title>", " ", ""],
            "sol": ["\npragma ", "\nusing ", "\ncontract ", "\ninterface ", "\nlibrary ",
                   "\nconstructor ", "\ntype ", "\nfunction ", "\nevent ", "\nmodifier ",
                   "\nerror ", "\nstruct ", "\nenum ", "\nif ", "\nfor ", "\nwhile ",
                   "\ndo while ", "\nassembly ", "\n\n", "\n", " ", ""],
        }
        return separators_map.get(language.lower(), DEFAULT_SEPARATORS[:])


def get_separators_for_file(file_path: str) -> List[str]:
    """Get appropriate separators based on file extension."""
    extension = Path(file_path).suffix.lstrip('.')
    return Language.get_separators(extension)


class RecursiveCharacterTextSplitter:
    """
    Text splitter that recursively tries different separators.
    Ported from aichat's Rust implementation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
            length_function: Function to measure text length (defaults to len())

        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        # Validate chunk parameters to prevent infinite loop in _merge_splits
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}). "
                f"This would cause an infinite loop during text splitting."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else DEFAULT_SEPARATORS[:]
        self.length_function = length_function if length_function is not None else len

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # Check if we should keep separators (any non-whitespace separator exists)
        keep_separator = any(
            any(not c.isspace() for c in sep)
            for sep in self.separators
        )
        return self._split_text_impl(text, self.separators, keep_separator, depth=0)

    def _split_text_impl(
        self,
        text: str,
        separators: List[str],
        keep_separator: bool,
        depth: int = 0,
        max_depth: int = 100
    ) -> List[str]:
        """
        Recursive implementation of text splitting with depth limiting.

        Args:
            text: Text to split
            separators: List of separators to try
            keep_separator: Whether to keep separators in output
            depth: Current recursion depth
            max_depth: Maximum allowed recursion depth

        Returns:
            List of text chunks

        Raises:
            RuntimeError: If recursion depth exceeds max_depth
        """
        # Prevent stack overflow from pathological inputs
        if depth > max_depth:
            raise RuntimeError(
                f"Text chunking exceeded maximum recursion depth ({max_depth}). "
                f"This may indicate an issue with the input text or separator configuration."
            )

        final_chunks = []

        # Find the appropriate separator
        separator = separators[-1] if separators else ""
        new_separators = []

        for i, sep in enumerate(separators):
            if not sep:  # Empty separator
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split on the separator
        splits = self._split_on_separator(text, separator, keep_separator)

        # Merge splits and recursively split larger texts
        good_splits = []
        _separator = "" if keep_separator else separator

        for s in splits:
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # Merge collected splits first
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []

                # Recursively split the large chunk
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text_impl(s, new_separators, keep_separator, depth=depth + 1)
                    final_chunks.extend(other_chunks)

        # Merge remaining splits
        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_on_separator(
        self,
        text: str,
        separator: str,
        keep_separator: bool
    ) -> List[str]:
        """Split text on separator, optionally keeping the separator."""
        if not separator:
            # Split into characters
            return [c for c in text if c]

        if keep_separator:
            # Keep separator at the start of each split
            splits = []
            prev_idx = 0
            sep_len = len(separator)

            while True:
                idx = text[prev_idx:].find(separator)
                if idx == -1:
                    break

                # Add chunk (with separator from previous split if applicable)
                start = max(0, prev_idx - sep_len)
                end = prev_idx + idx
                if text[start:end]:
                    splits.append(text[start:end])

                prev_idx = prev_idx + idx + sep_len

            # Add remaining text
            if prev_idx < len(text):
                start = max(0, prev_idx - sep_len)
                if text[start:]:
                    splits.append(text[start:])
        else:
            # Normal split without keeping separator
            splits = text.split(separator)

        # Filter out empty strings
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks, respecting chunk_size and chunk_overlap."""
        docs = []
        current_doc = []
        total = 0

        for d in splits:
            _len = self.length_function(d)
            separator_len = len(current_doc) * self.length_function(separator)

            if total + _len + separator_len > self.chunk_size:
                # Save current document
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc:
                        docs.append(doc)

                    # Keep removing from start while over overlap threshold
                    while (total > self.chunk_overlap or
                           (total + _len + len(current_doc) * self.length_function(separator) > self.chunk_size
                            and total > 0)):
                        total -= self.length_function(current_doc[0])
                        current_doc.pop(0)

            current_doc.append(d)
            total += _len

        # Add final document
        if current_doc:
            doc = self._join_docs(current_doc, separator)
            if doc:
                docs.append(doc)

        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join documents with separator, returning None for empty results."""
        text = separator.join(docs).strip()
        return text if text else None


def create_splitter_for_file(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with appropriate separators for the file type."""
    separators = get_separators_for_file(file_path)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
