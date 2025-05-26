import os
import tempfile
from typing import cast
from unittest.mock import Mock, patch

import pytest
from sqlite_utils import Database
from sqlite_utils.db import Table

from llm_tools_rag import RAG


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def db_with_collections(temp_db):
    db = Database(temp_db)
    cast(Table, db["collections"]).insert({"name": "collection1"})
    cast(Table, db["collections"]).insert({"name": "collection2"})
    return temp_db


def test_get_collections(db_with_collections):
    result = RAG().get_collections(database=db_with_collections)
    assert result == ["collection1", "collection2"]


def test_get_collections_no_table(temp_db):
    with pytest.raises(RuntimeError, match="No collections database found"):
        RAG().get_collections(database=temp_db)


@patch("llm_tools_rag.llm.Collection")
def test_get_relevant_documents(mock_collection_class, db_with_collections):
    mock_collection = Mock()
    mock_collection_class.return_value = mock_collection
    mock_entry = Mock()
    mock_collection.similar.return_value = [mock_entry]

    with patch("llm_tools_rag.asdict", return_value={"id": "1", "content": "test"}):
        result = RAG().get_relevant_documents(
            "query", "collection1", db_with_collections, number=5
        )

        mock_collection.similar.assert_called_once_with("query", number=5)
        assert result == [{"id": "1", "content": "test"}]


def test_get_relevant_documents_no_table(temp_db):
    with pytest.raises(RuntimeError, match="No collections database found"):
        RAG().get_relevant_documents("query", "collection1", temp_db)
