"""
Advanced RAG plugin for LLM with hybrid search (ChromaDB + BM25).
"""

import llm
from typing import Optional, List, Dict, Any
import click

from .engine import RAGEngine, get_or_create_engine
from .config import list_collections, get_rag_config_dir
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


@llm.hookimpl
def register_tools(register):
    """Register RAG as an LLM tool."""

    @register
    class RAGTool:
        """Search indexed document collections using hybrid semantic+keyword search."""

        name = "rag"

        def __init__(self, collection: str, top_k: int = 5, mode: str = "hybrid"):
            """
            Args:
                collection: Name of the RAG collection to search (REQUIRED)
                top_k: Number of results to return (default: 5)
                mode: Search mode - "hybrid" (default), "vector", or "keyword"
            """
            self.collection = collection
            self.top_k = top_k
            self.mode = mode

        def __call__(self, query: str) -> str:
            """
            Search indexed documents. Use when user asks to search their docs/codebase.

            IMPORTANT: Do NOT use this tool unless the user explicitly asks to:
            - "search my docs for...", "what do my notes say about..."
            - "find in my codebase...", "where is X defined?"
            - "look up in [collection name]..."

            Do NOT use for:
            - General knowledge questions you can answer from training data
            - Web searches (use search_google instead)
            - Real-time/current information not in indexed documents

            Args:
                query: Search query - be specific for better results.
                       Examples: "authentication flow", "def search_", "RAGEngine"

            Returns:
                Numbered document excerpts with source paths, or "No relevant documents found."
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


__version__ = "0.1.0"
