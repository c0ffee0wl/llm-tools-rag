# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
llm install -e .

# Run tests
pytest tests/ -v

# Run single test
pytest tests/test_search.py::test_rrf_basic -v

# Manual integration test
llm rag add test git:/opt/llm-tools-rag && llm rag search test "hybrid search" && llm rag delete test

# Test as LLM tool
llm --tool rag --tool-option collection test "query"
```

## Architecture

**Hybrid RAG plugin for llm CLI** combining ChromaDB (vector/HNSW) + BM25 (keyword) via Reciprocal Rank Fusion.

```
llm_tools_rag/
├── __init__.py   # Plugin registration (@llm.hookimpl for CLI commands + RAGTool)
├── engine.py     # RAGEngine: orchestrates ChromaDB + BM25, handles embedding, caching
├── search.py     # HybridSearch: BM25Index + RRF algorithm
├── chunking.py   # RecursiveCharacterTextSplitter with language-aware separators
├── loaders.py    # DocumentLoader: protocol handlers (git:, pdf, docx, URL crawling)
├── config.py     # RAGConfig singleton, collection validation, deep merge
├── dedup.py      # SHA256 deduplication (hash-only, no content storage)
└── repl.py       # Interactive REPL mode
```

### Data Flow

```
Document → loaders.py (protocol detection) → chunking.py (split) → dedup.py (filter)
    → engine.py (embed via llm.get_embedding_model())
    → ChromaDB (vectors) + BM25Index (keywords)

Query → engine.py (embed) → ChromaDB search + BM25 search → search.py (RRF fusion) → results
```

### Storage

```
~/.config/io.datasette.llm/rag/<collection>/
├── chromadb/chroma.sqlite3    # Vectors + metadata
├── bm25_cache.pkl             # HMAC-signed BM25 index cache
└── .lock                      # flock for concurrency
```

## Key Implementation Details

**ID Mapping**: BM25 uses integer IDs, ChromaDB uses strings. `engine.py` maintains bidirectional mapping (`_id_to_bm25_id`, `_bm25_id_to_id`).

**BM25 Caching**: Pickled to disk with HMAC signature. Invalidated when chunk count changes.

**Embedding Format**: ChromaDB queries require `[[embedding]]` (list of lists), not `[embedding]`.

**RRF Formula** (search.py): `score = weight / (rrf_k + rank + 1)` - weights normalized to sum to 1.0.

**Locking**: `fcntl.flock()` with timeout for concurrent access to collections.

## Critical Invariants

1. **Embedding model must be consistent** across all documents in a collection
2. **chunk_overlap < chunk_size** (enforced in config.py and chunking.py)
3. **Collection names**: alphanumeric, hyphens, underscores only (path traversal protection)
4. **Subprocess calls**: Always `shell=False` with `shlex.split()` (injection protection)

## Config Reference

```yaml
embedding_model: null          # Uses llm default if not set
chunk_size: 1000               # Characters, not tokens
chunk_overlap: 100
top_k: 5
search_mode: "hybrid"          # vector | keyword | hybrid
vector_weight: 0.7
keyword_weight: 0.3
document_loaders:
  git: "yek $1"
  pdf: "pdftotext $1 -"
  docx: "pandoc --to plain $1"
```

## Adding a Document Loader

In `loaders.py` DEFAULT_LOADERS or per-collection config.yaml:
```python
"myformat": "my-converter $1"  # $1 = input path, output to stdout
```

## Debugging

```python
# Check ChromaDB
import chromadb
client = chromadb.PersistentClient(path="~/.config/io.datasette.llm/rag/mycoll/chromadb")
coll = client.get_collection("mycoll")
print(coll.count(), coll.get(limit=5))

# Check BM25
from llm_tools_rag.engine import get_or_create_engine
engine = get_or_create_engine("mycoll")
print(engine.hybrid_search.bm25_index.size())
```
