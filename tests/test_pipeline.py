"""
Test suite for RAG pipeline logic
"""

from pipeline import RAGPipeline, get_pipeline
from langchain_core.documents import Document


def test_clear_all_resets_store():
    """Clear all should remove any previously ingested documents"""
    pipeline = RAGPipeline()
    # make sure there's at least one document so we can clear it
    pipeline.vector_store.clear()
    pipeline.vector_store.add_documents([Document(page_content="hello", metadata={"source": "x"})])
    before = pipeline.get_stats()["vector_store"]["document_count"]
    assert before > 0
    assert pipeline.clear_all() is True
    after = pipeline.get_stats()["vector_store"]["document_count"]
    assert after == 0


def test_ingest_uploaded_files_empty():
    """Uploading an empty list should return a failure result"""
    pipeline = RAGPipeline()
    result = pipeline.ingest_uploaded_files([], clear_existing=True)
    assert result["success"] is False
    assert result["documents_loaded"] == 0


def test_get_pipeline_singleton():
    """get_pipeline should always return the same instance"""
    first = get_pipeline()
    second = get_pipeline()
    assert first is second
