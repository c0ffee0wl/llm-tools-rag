from dataclasses import asdict

import llm
from llm import user_dir
from sqlite_utils import Database


def get_collections(database: str | None = None):
    """Retrieve all collection names from the embeddings database.

    Args:
        database: Path to the embeddings database file. If None, uses the default
                 embeddings.db file in the user directory.

    Returns:
        list[str]: A list of collection names.

    Raises:
        RuntimeError: If no collections table exists in the database.
    """
    if database is None:
        database = str(user_dir() / "embeddings.db")
    db = Database(database)
    if not db["collections"].exists():
        raise RuntimeError("No collections database found in {database}")
    rows = db.query("SELECT collections.name FROM collections")
    return [row["name"] for row in rows]


def get_relevant_documents(
    query: str, collection_name: str, database: str | None = None, number: int = 3
) -> list[dict]:
    """Find items in a collection that are similar to the given query.

    Args:
        query: The text to find similar embeddings for
        collection_name: Name of the collection to search in
        database: Path to the embeddings database file. If None, uses the default
                 embeddings.db file in the user directory.
        number: Maximum number of similar items to return (default: 10)

    Returns:
        list[dict]: A list of dictionaries containing id, score, content (if stored),
                   and metadata (if stored) for similar items.

    Raises:
        RuntimeError: If the specified collection doesn't exist in the database.
    """
    if database is None:
        database = str(user_dir() / "embeddings.db")

    db = Database(database)

    # Check if collection exists
    if not db["collections"].exists():
        raise RuntimeError(f"No collections database found in {database}")

    # Get the collection and perform similarity search
    collection = llm.Collection(collection_name, db)

    return [asdict(entry) for entry in collection.similar(query, number=number)]


@llm.hookimpl
def register_tools(register):
    register(get_collections)
    register(get_relevant_documents)
