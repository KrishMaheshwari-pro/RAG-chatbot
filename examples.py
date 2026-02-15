"""
Example usage and quickstart guide
"""

from pipeline import get_pipeline
from pathlib import Path


def example_basic_usage():
    """
    Basic example: Load documents, create embeddings, and query
    """
    print("Initializing RAG Pipeline...")
    pipeline = get_pipeline()
    
    print("\nLoading documents from data/documents/...")
    result = pipeline.ingest_documents(
        str(Path("data/documents")),
        clear_existing=True
    )
    
    if result["success"]:
        print(f"✓ {result['message']}")
    else:
        print(f"✗ Error: {result['message']}")
        return
    
    print("\nAsking a question...")
    query = "What is the main topic of the documents?"
    response = pipeline.query(query)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {response['answer']}")
    
    if response.get("citations"):
        print("\nSources:")
        for cite in response["citations"]:
            print(f"  - {cite['source']} (relevance: {cite.get('similarity', 0):.0%})")


def example_upload_files():
    """
    Example: Upload and process custom files
    """
    from pathlib import Path
    
    pipeline = get_pipeline()
    
    # Example file paths
    files = [
        "path/to/document1.pdf",
        "path/to/document2.txt"
    ]
    
    result = pipeline.ingest_uploaded_files(
        files,
        clear_existing=False
    )
    
    print(result["message"])


if __name__ == "__main__":
    example_basic_usage()
