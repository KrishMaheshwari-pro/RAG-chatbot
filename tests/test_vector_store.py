"""
Test suite for vector store module
"""

import pytest
from vector_store import get_vector_store
from langchain_core.documents import Document


def test_vector_store_creation():
    """Test vector store initialization"""
    store = get_vector_store()
    assert store is not None


def test_add_documents():
    """Test adding documents to vector store"""
    store = get_vector_store()
    # clear any existing data so the test is idempotent
    store.clear()
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test.txt"})
    ]
    result = store.add_documents(docs)
    assert result == len(docs)
    stats = store.get_stats()
    assert stats["document_count"] == len(docs)


def test_query_method():
    """Test semantic search functionality uses `query` API"""
    store = get_vector_store()
    query = "test query"
    results = store.query(query, top_k=1)
    assert isinstance(results, list)
