from dataclasses import asdict
from typing import Optional

import llm
from llm import user_dir
from sqlite_utils import Database


class RAG(llm.Toolbox):
    name: str = "RAG"

    def __init__(self, database: Optional[str] = None):
        self.database = (
            database if database is not None else str(user_dir() / "embeddings.db")
        )

    def get_collections(self):
        """Retrieve all collection names from the embeddings database.

        Returns:
            list[str]: A list of collection names.

        Raises:
            RuntimeError: If no collections table exists in the database.
        """
        db = Database(self.database)
        if not db["collections"].exists():
            raise RuntimeError(f"No collections database found in {self.database}")
        rows = db.query("SELECT collections.name FROM collections")
        return [row["name"] for row in rows]

    def get_relevant_documents(
        self,
        query: str,
        collection_name: str,
        number: int = 3,
    ) -> list[dict]:
        """Find items in a collection that are similar to the given query.

        Args:
            query: The text to find similar embeddings for
            collection_name: Name of the collection to search in
            number: Maximum number of similar items to return (default: 10)

        Returns:
            list[dict]: A list of dictionaries containing id, score, content (if stored),
                    and metadata (if stored) for similar items.

        Raises:
            RuntimeError: If the specified collection doesn't exist in the database.
        """
        db = Database(self.database)

        # Check if collection exists
        if not db["collections"].exists():
            raise RuntimeError(f"No collections database found in {self.database}")

        # Get the collection and perform similarity search
        collection = llm.Collection(collection_name, db)

        return [asdict(entry) for entry in collection.similar(query, number=number)]


@llm.hookimpl
def register_tools(register):
    register(RAG)
