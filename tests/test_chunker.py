"""
Test suite for document chunking module
"""

import pytest
from chunker import chunk_documents
from langchain_core.documents import Document


def test_chunk_documents():
    """Test document chunking"""
    docs = [
        Document(page_content="This is a long document. " * 100, metadata={"source": "test.txt"})
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunk_size():
    """Test chunk size constraints"""
    docs = [
        Document(page_content="A " * 500, metadata={"source": "test.txt"})
    ]
    chunks = chunk_documents(docs)
    assert all(len(chunk.page_content) > 0 for chunk in chunks)


def test_metadata_preservation():
    """Test metadata preservation during chunking"""
    docs = [
        Document(page_content="Test content " * 100, metadata={"source": "test.pdf", "page": 1})
    ]
    chunks = chunk_documents(docs)
    assert all("source" in chunk.metadata for chunk in chunks)
