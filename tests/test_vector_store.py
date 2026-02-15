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
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test.txt"})
    ]
    result = store.add_documents(docs)
    assert result is not None


def test_similarity_search():
    """Test semantic search functionality"""
    store = get_vector_store()
    query = "test query"
    results = store.similarity_search(query, k=1)
    assert isinstance(results, list)
