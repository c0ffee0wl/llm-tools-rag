"""
Advanced RAG plugin for LLM with hybrid search (ChromaDB + BM25).
"""

import llm
from typing import Optional, List, Dict, Any
import click

from .engine import RAGEngine, get_or_create_engine
from .config import list_collections, get_rag_config_dir, DEFAULT_CONFIG
from .loaders import check_loader_dependencies, get_missing_dependencies
from .repl import start_repl


@llm.hookimpl
def register_commands(cli):
    """Register the 'llm rag' command."""

    @cli.group(name="rag")
    def rag_group():
        """RAG (Retrieval-Augmented Generation) with hybrid search."""
        pass

    @rag_group.command(name="list")
    def list_cmd():
        """List all RAG collections."""
        collections = list_collections()
        if not collections:
            click.echo("No RAG collections found.")
            return

        click.echo(f"RAG collections ({len(collections)}):")
        for name in collections:
            click.echo(f"  - {name}")

    @rag_group.command(name="add")
    @click.argument("collection")
    @click.argument("path")
    @click.option("--refresh", is_flag=True, help="Reindex even if document exists")
    @click.option("--model", "-m", help="Embedding model to use")
    def add_cmd(collection: str, path: str, refresh: bool, model: Optional[str]):
        """Add a document to a RAG collection."""
        engine = get_or_create_engine(collection, model)
        result = engine.add_document(path, refresh=refresh)

        if result["status"] == "success":
            click.echo(f"✓ Added {path} ({result['chunks']} chunks)")
        elif result["status"] == "skipped":
            click.echo(f"⊘ Skipped {path}: {result['reason']}")
        else:
            click.echo(f"✗ Error adding {path}: {result.get('error', 'unknown')}", err=True)

    @rag_group.command(name="search")
    @click.argument("collection")
    @click.argument("query")
    @click.option("--top-k", "-k", type=int, help="Number of results")
    @click.option("--mode", type=click.Choice(["vector", "keyword", "hybrid"]), help="Search mode")
    def search_cmd(collection: str, query: str, top_k: Optional[int], mode: Optional[str]):
        """Search a RAG collection."""
        try:
            engine = get_or_create_engine(collection)
            results = engine.search(query, top_k=top_k, mode=mode)

            if not results:
                click.echo("No results found.")
                return

            for i, result in enumerate(results, 1):
                click.echo(f"\n--- Result {i} ---")
                click.echo(f"Source: {result['metadata'].get('source', 'unknown')}")
                click.echo(f"Content:\n{result['content'][:500]}...")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    @rag_group.command(name="info")
    @click.argument("collection")
    def info_cmd(collection: str):
        """Show information about a RAG collection."""
        try:
            engine = get_or_create_engine(collection)
            stats = engine.get_stats()

            click.echo(f"Collection: {stats['collection']}")
            click.echo(f"Total chunks: {stats['total_chunks']}")
            click.echo(f"Unique documents: {stats['unique_documents']}")
            click.echo(f"\nConfiguration:")
            for key, value in stats['config'].items():
                if key != "document_loaders":
                    click.echo(f"  {key}: {value}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    @rag_group.command(name="delete")
    @click.argument("collection")
    @click.confirmation_option(prompt="Are you sure you want to delete this collection?")
    def delete_cmd(collection: str):
        """Delete a RAG collection."""
        try:
            engine = get_or_create_engine(collection)
            engine.delete_collection()
            click.echo(f"✓ Deleted collection: {collection}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    @rag_group.command(name="rebuild")
    @click.argument("collection")
    def rebuild_cmd(collection: str):
        """Rebuild BM25 index for a collection."""
        try:
            engine = get_or_create_engine(collection)
            engine.rebuild_index()
            click.echo(f"✓ Rebuilt index for: {collection}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    @rag_group.command(name="check-deps")
    def check_deps_cmd():
        """Check document loader dependencies."""
        status = check_loader_dependencies()
        missing = get_missing_dependencies()

        click.echo("Document loader status:")
        for loader_type, available in status.items():
            symbol = "✓" if available else "✗"
            click.echo(f"  {symbol} {loader_type}")

        if missing:
            click.echo("\nMissing dependencies:")
            for loader_type, install_cmd in missing.items():
                click.echo(f"  {loader_type}: {install_cmd}")

    @rag_group.command(name="sources")
    @click.argument("collection")
    def sources_cmd(collection: str):
        """Show source documents from last search in this collection."""
        try:
            engine = get_or_create_engine(collection)
            sources = engine.get_last_sources()

            if not sources:
                click.echo("No recent search results. Run 'llm rag search' first.")
                return

            click.echo(f"Sources used in last search ({len(sources)}):")
            for source in sources:
                click.echo(f"  - {source}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    @rag_group.command(name="documents")
    @click.argument("collection")
    def documents_cmd(collection: str):
        """List all documents in a collection."""
        try:
            engine = get_or_create_engine(collection)
            documents = engine.list_documents()

            if not documents:
                click.echo(f"No documents in collection '{collection}'.")
                return

            click.echo(f"Documents in '{collection}' ({len(documents)}):")
            for doc in documents:
                click.echo(f"  - {doc}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    @rag_group.command(name="repl")
    @click.argument("collection", required=False)
    def repl_cmd(collection: Optional[str]):
        """Start interactive REPL mode."""
        start_repl(collection)


class RAGTool(llm.Toolbox):
    """
    Search indexed document collections using hybrid semantic+keyword search.

    RAG (Retrieval-Augmented Generation) toolbox for searching pre-indexed documents.
    Collections are created and populated using 'llm rag add'. Supports multiple search
    modes: hybrid (default), vector-only, or keyword-only.
    """

    name = "rag"

    def __init__(
        self,
        collection: str,
        top_k: int = DEFAULT_CONFIG["top_k"],
        mode: str = DEFAULT_CONFIG["search_mode"]
    ):
        """
        Initialize RAG search for a specific collection.

        Args:
            collection: Name of the RAG collection to search (REQUIRED).
                        Use 'llm rag list' to see available collections.
            top_k: Number of results to return
            mode: Search mode - "hybrid", "vector", or "keyword"
        """
        self.collection = collection
        self.top_k = top_k
        self.mode = mode

    def search(self, query: str) -> str:
        """
        Search indexed documents using hybrid semantic and keyword matching.

        Searches the configured RAG collection and returns relevant document excerpts
        with source attribution. Results are ranked by a combination of vector similarity
        and BM25 keyword matching for comprehensive retrieval.

        Args:
            query: Search query - be specific for better results.
                   Examples: "authentication flow", "def search_", "RAGEngine",
                   "how does X work", "error handling in module Y"

        Returns:
            Numbered document excerpts with source file paths, formatted for context.
            Returns "No relevant documents found." if no matches.
        """
        try:
            engine = get_or_create_engine(self.collection)
            results = engine.search(query, top_k=self.top_k, mode=self.mode)

            if not results:
                return "No relevant documents found."

            # Format results for LLM consumption
            formatted = []
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'unknown')
                content = result['content']
                formatted.append(f"[Document {i} - {source}]\n{content}")

            return "\n\n".join(formatted)

        except Exception as e:
            return f"Error searching RAG collection: {e}"


@llm.hookimpl
def register_tools(register):
    """Register RAG as an LLM tool."""
    register(RAGTool)


__version__ = "0.1.0"

# =============================================================================
# Library API - High-level functions for external consumers (e.g., llm-assistant)
# =============================================================================

__all__ = [
    # Tool class
    'RAGTool',
    # Library API for external consumers
    'search_collection',
    'add_to_collection',
    'get_collection_list',
    'collection_exists',
    'get_collection_stats',
    'rebuild_collection_index',
    'remove_collection',
]


def search_collection(
    collection: str,
    query: str,
    top_k: int = DEFAULT_CONFIG["top_k"],
    mode: str = DEFAULT_CONFIG["search_mode"]
) -> List[Dict[str, Any]]:
    """
    Search a RAG collection. Returns list of results.

    Args:
        collection: Name of the collection to search
        query: Search query string
        top_k: Number of results to return
        mode: Search mode - "hybrid", "vector", or "keyword"

    Returns:
        List of result dicts with: id, content, metadata (source, chunk_index, etc.)
    """
    engine = get_or_create_engine(collection)
    return engine.search(query, top_k=top_k, mode=mode)


def add_to_collection(
    collection: str,
    path: str,
    refresh: bool = False,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add document(s) to a collection. Creates collection if needed.

    Args:
        collection: Collection name
        path: File path, git:url, or glob pattern
        refresh: Force reindex if document exists
        model: Optional embedding model override

    Returns:
        Dict with status, chunks count, errors
    """
    engine = get_or_create_engine(collection, model)
    return engine.add_document(path, refresh=refresh)


def get_collection_list() -> List[Dict[str, Any]]:
    """
    List all collections with metadata.

    Returns:
        List of dicts with: name, chunks, documents
    """
    names = list_collections()
    result = []
    for name in names:
        try:
            engine = get_or_create_engine(name)
            stats = engine.get_stats()
            result.append({
                'name': name,
                'chunks': stats['total_chunks'],
                'documents': stats['unique_documents']
            })
        except Exception:
            result.append({'name': name, 'chunks': '?', 'documents': '?'})
    return result


def collection_exists(name: str) -> bool:
    """Check if a collection exists."""
    return name in list_collections()


def get_collection_stats(collection: str) -> Dict[str, Any]:
    """Get detailed statistics for a collection."""
    engine = get_or_create_engine(collection)
    return engine.get_stats()


def rebuild_collection_index(collection: str) -> None:
    """Rebuild a collection's BM25 index."""
    engine = get_or_create_engine(collection)
    engine.rebuild_index()


def remove_collection(collection: str) -> None:
    """Delete a collection and all its data."""
    engine = get_or_create_engine(collection)
    engine.delete_collection()
