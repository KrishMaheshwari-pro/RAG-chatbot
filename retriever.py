"""
Retriever Module
Semantic search to find relevant document chunks
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import TOP_K, SIMILARITY_THRESHOLD
from vector_store import get_vector_store


class Retriever:
    """
    Retrieves relevant document chunks based on semantic similarity.
    """
    
    def __init__(
        self,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):
        """
        Initialize the retriever.
        
        Args:
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_store = get_vector_store()
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Override default number of results
            filter_sources: Optional list of source files to filter by
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        k = top_k or self.top_k
        
        # Build metadata filter if needed
        filter_metadata = None
        if filter_sources:
            # ChromaDB where filter for source files
            if len(filter_sources) == 1:
                filter_metadata = {"source": filter_sources[0]}
            else:
                filter_metadata = {"source": {"$in": filter_sources}}
        
        # Query vector store
        results = self.vector_store.query(
            query_text=query,
            top_k=k,
            filter_metadata=filter_metadata
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r.get("similarity", 0) >= self.similarity_threshold
        ]
        
        if len(filtered_results) < len(results):
            print(f"[*] Filtered {len(results) - len(filtered_results)} low-relevance results")
        
        return filtered_results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Retrieve relevant documents and format them as context.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            
        Returns:
            Formatted context string with citations
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            page = result["metadata"].get("page", "")
            page_info = f", Page {page + 1}" if page != "" else ""
            
            context_parts.append(
                f"[{i}] Source: {source}{page_info}\n"
                f"{result['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique sources from results for citation.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of unique sources with citation numbers
        """
        seen_sources = {}
        sources = []
        
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            page = result["metadata"].get("page", "")
            
            key = f"{source}_{page}"
            if key not in seen_sources:
                seen_sources[key] = i
                sources.append({
                    "citation": i,
                    "source": source,
                    "page": page + 1 if isinstance(page, int) else page,
                    "similarity": result.get("similarity", 0)
                })
        
        return sources


# Singleton instance
_retriever = None


def get_retriever() -> Retriever:
    """Get the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


if __name__ == "__main__":
    # Test retriever
    retriever = Retriever()
    
    # This will work only if documents have been ingested
    results = retriever.retrieve("What is this document about?")
    
    if results:
        print("\n[*] Retrieved Documents:")
        for r in results:
            print(f"\nScore: {r['similarity']:.3f}")
            print(f"Source: {r['metadata'].get('source', 'Unknown')}")
            print(f"Content: {r['content'][:150]}...")
    else:
        print("No documents found. Please ingest documents first.")
