"""
Test suite for embeddings module
"""

import pytest
from embeddings import get_embedding_model


def test_get_embedding_model():
    """Test embedding model initialization"""
    model = get_embedding_model()
    assert model is not None


def test_embedding_dimension():
    """Test embedding output dimensions"""
    model = get_embedding_model()
    texts = ["Hello world", "This is a test"]
    embeddings = model.embed_documents(texts)
    assert len(embeddings) == 2
    assert all(len(e) > 0 for e in embeddings)


def test_embedding_similarity():
    """Test semantic similarity calculation"""
    model = get_embedding_model()
    query = "What is AI?"
    query_embedding = model.embed_query(query)
    assert len(query_embedding) > 0
