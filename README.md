# llm-tools-rag

Advanced RAG (Retrieval-Augmented Generation) plugin for [llm](https://github.com/simonw/llm) with hybrid search capabilities.

## Features

- **Hybrid Search**: Combines vector similarity (HNSW) with keyword search (BM25) using Reciprocal Rank Fusion
- **High Performance**: ChromaDB's 2025 Rust-core for fast vector operations
- **Protocol-Based Loaders**: Support for `git:`, PDF, DOCX, and more
- **Smart Chunking**: Language-aware text splitting with configurable size and overlap
- **Deduplication**: SHA256-based content deduplication
- **Per-Collection Config**: YAML-based configuration per RAG collection

## Installation

```bash
# Install via pip
pip install llm-tools-rag

# Or install in development mode
cd llm-tools-rag
llm install -e .
```

## Usage

### Command-Line Operations

```bash
# Add documents to collection
llm rag add my-docs git:https://github.com/simonw/llm
llm rag add my-docs ~/documents/manual.pdf

# Search collection
llm rag search my-docs "how do I install plugins?"

# List documents
llm rag info my-docs
```

### Tool Mode (within LLM conversation)

```bash
llm --tool rag --tool-option collection my-docs "explain the plugin system"
```

### Additional Commands

```bash
# Rebuild BM25 index
llm rag rebuild my-docs

# Delete collection
llm rag delete my-docs

# List all collections
llm rag list

# Check loader dependencies
llm rag check-deps
```

## Document Loaders

Supports protocol-based loading:

- `git:/path/or/url` - Use yek to extract repository contents
- `*.pdf` - Extract text via pdftotext
- `*.docx` - Extract text via pandoc
- Plain text files - Direct reading

## Configuration

Per-collection config stored in `~/.config/io.datasette.llm/rag/<collection>/config.yaml`:

```yaml
embedding_model: "azure/text-embedding-3-small"
chunk_size: 1000
chunk_overlap: 100
top_k: 5
search_mode: "hybrid"  # vector | keyword | hybrid
rrf_k: 60
vector_weight: 0.7
keyword_weight: 0.3
document_loaders:
  git: "yek $1"
  pdf: "pdftotext $1 -"
  docx: "pandoc --to plain $1"
```

## Architecture

- **Vector Store**: ChromaDB with HNSW indexing
- **Keyword Search**: rank-bm25 for BM25 scoring
- **Fusion**: Reciprocal Rank Fusion (RRF) to combine rankings
- **Storage**: Persistent collections with metadata

## License

Apache-2.0
