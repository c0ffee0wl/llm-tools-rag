"""
Tests for text chunking functionality.
"""

import pytest
from llm_tools_rag.chunking import RecursiveCharacterTextSplitter, Language


def test_split_text_basic():
    """Test basic text splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=7,
        chunk_overlap=3,
        separators=[" "]
    )
    output = splitter.split_text("foo bar baz 123")
    assert output == ["foo bar", "bar baz", "baz 123"]


def test_split_text_no_overlap():
    """Test splitting without overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3,
        chunk_overlap=0,
        separators=[" "]
    )
    output = splitter.split_text("foo bar baz")
    assert "foo" in output
    assert "bar" in output
    assert "baz" in output


def test_markdown_splitter():
    """Test Markdown-aware splitting."""
    text = """# Title

Some text here.

## Section

More text here."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=Language.get_separators("md")
    )
    output = splitter.split_text(text)

    # Should preserve headings
    assert any("# Title" in chunk for chunk in output)
    assert any("## Section" in chunk for chunk in output)


def test_python_splitter():
    """Test Python-aware splitting."""
    code = """class Foo:
    def method1(self):
        pass

def function():
    x = 1
    return x"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=0,
        separators=Language.get_separators("py")
    )
    output = splitter.split_text(code)

    # Should try to preserve class and function definitions
    assert len(output) > 0


def test_empty_text():
    """Test handling of empty text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    output = splitter.split_text("")
    assert output == []


def test_text_smaller_than_chunk():
    """Test text that's smaller than chunk size."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    output = splitter.split_text("Small text")
    assert output == ["Small text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
