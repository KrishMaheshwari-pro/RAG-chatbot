"""
Test suite for retriever module
"""

import pytest
from retriever import get_retriever
from langchain_core.documents import Document


def test_retriever_creation():
    """Test retriever initialization"""
    retriever = get_retriever()
    assert retriever is not None


def test_retrieve_documents():
    """Test document retrieval"""
    retriever = get_retriever()
    query = "sample query"
    results = retriever.retrieve(query)
    assert isinstance(results, list)
