"""
Interactive REPL mode for RAG collections.
Provides commands similar to aichat's .rag interface.
"""

import cmd
from typing import Optional
from .engine import RAGEngine, get_or_create_engine
from .config import list_collections


class RAGRepl(cmd.Cmd):
    """Interactive REPL for RAG operations."""

    intro = "RAG Interactive Mode. Type 'help' or '?' for commands.\n"
    prompt = "rag> "

    def __init__(self, collection: Optional[str] = None):
        """
        Initialize REPL.

        Args:
            collection: Optional collection to load on startup
        """
        super().__init__()
        self.engine: Optional[RAGEngine] = None
        self.collection_name: Optional[str] = None

        if collection:
            self.do_use(collection)

    def do_use(self, collection: str):
        """
        Load or create a RAG collection.

        Usage: use <collection_name>
        """
        if not collection:
            print("Error: Please specify a collection name")
            return

        try:
            self.engine = get_or_create_engine(collection)
            self.collection_name = collection
            self.prompt = f"rag:{collection}> "
            print(f"Loaded collection: {collection}")
        except Exception as e:
            print(f"Error loading collection: {e}")

    def do_list(self, arg):
        """
        List all RAG collections.

        Usage: list
        """
        collections = list_collections()
        if not collections:
            print("No collections found.")
        else:
            print(f"Collections ({len(collections)}):")
            for name in collections:
                marker = " *" if name == self.collection_name else ""
                print(f"  - {name}{marker}")

    def do_add(self, path: str):
        """
        Add a document to the current collection.

        Usage: add <path>
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        if not path:
            print("Error: Please specify a document path")
            return

        print(f"Adding document: {path}")
        result = self.engine.add_document(path)

        if result["status"] == "success":
            print(f"✓ Added ({result['chunks']} chunks)")
        elif result["status"] == "skipped":
            print(f"⊘ Skipped: {result['reason']}")
        else:
            print(f"✗ Error: {result.get('error', 'unknown')}")

    def do_search(self, query: str):
        """
        Search the current collection.

        Usage: search <query>
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        if not query or not query.strip():
            print("Error: Please specify a non-empty search query")
            return

        try:
            results = self.engine.search(query)

            if not results:
                print("No results found.")
                return

            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Source: {result['metadata'].get('source', 'unknown')}")
                print(f"Content:\n{result['content'][:500]}...")
        except Exception as e:
            print(f"Error: {e}")

    def do_sources(self, arg):
        """
        Show source documents from the last search.

        Usage: sources
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        sources = self.engine.get_last_sources()

        if not sources:
            print("No recent search results.")
            return

        print(f"Sources ({len(sources)}):")
        for source in sources:
            print(f"  - {source}")

    def do_documents(self, arg):
        """
        List all documents in the current collection.

        Usage: documents
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        documents = self.engine.list_documents()

        if not documents:
            print("No documents in collection.")
            return

        print(f"Documents ({len(documents)}):")
        for doc in documents:
            print(f"  - {doc}")

    def do_info(self, arg):
        """
        Show information about the current collection.

        Usage: info
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        try:
            stats = self.engine.get_stats()

            print(f"Collection: {stats['collection']}")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Unique documents: {stats['unique_documents']}")
            print("\nConfiguration:")
            for key, value in stats['config'].items():
                if key != "document_loaders":
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error: {e}")

    def do_rebuild(self, arg):
        """
        Rebuild the BM25 index for the current collection.

        Usage: rebuild
        """
        if not self.engine:
            print("Error: No collection loaded. Use 'use <collection>' first.")
            return

        try:
            self.engine.rebuild_index()
            print("✓ Index rebuilt successfully")
        except Exception as e:
            print(f"Error: {e}")

    def do_exit(self, arg):
        """
        Exit the REPL.

        Usage: exit
        """
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """
        Exit the REPL (alias for exit).

        Usage: quit
        """
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Handle Ctrl+D."""
        print("\nGoodbye!")
        return True

    def emptyline(self):
        """Do nothing on empty line."""
        pass


def start_repl(collection: Optional[str] = None):
    """
    Start the interactive REPL.

    Args:
        collection: Optional collection to load on startup
    """
    repl = RAGRepl(collection)
    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
