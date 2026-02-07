"""
RAG Pipeline Module
End-to-end orchestration of document ingestion and query processing
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import DOCUMENTS_DIR, TOP_K
from document_loader import load_documents
from chunker import chunk_documents
from vector_store import get_vector_store, VectorStore
from retriever import get_retriever, Retriever
from generator import get_generator, Generator


class RAGPipeline:
    """
    End-to-end RAG pipeline for document Q&A.
    Handles document ingestion, retrieval, and response generation.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Optional custom vector store
            retriever: Optional custom retriever
            generator: Optional custom generator
        """
        self.vector_store = vector_store or get_vector_store()
        self.retriever = retriever or get_retriever()
        self.generator = generator or get_generator()
        self._documents_ingested = False
    
    def ingest_documents(
        self,
        source_path: str = None,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest documents from a directory or file.
        
        Args:
            source_path: Path to documents (defaults to DOCUMENTS_DIR)
            clear_existing: Whether to clear existing documents first
            
        Returns:
            Ingestion statistics
        """
        path = source_path or str(DOCUMENTS_DIR)
        
        print(f"\n{'='*50}")
        print(f"[*] DOCUMENT INGESTION")
        print(f"{'='*50}")
        print(f"Source: {path}")
        
        # Clear existing if requested
        if clear_existing:
            print("[*] Clearing existing documents...")
            self.vector_store.clear()
        
        # Load documents
        print("\n[*] Loading documents...")
        documents = load_documents(path)
        
        if not documents:
            return {
                "success": False,
                "documents_loaded": 0,
                "chunks_created": 0,
                "message": "No documents found to ingest"
            }
        
        # Chunk documents
        print("\n[*] Chunking documents...")
        chunks = chunk_documents(documents)
        
        # Add to vector store
        print("\n[*] Storing embeddings...")
        added = self.vector_store.add_documents(chunks)
        
        self._documents_ingested = True
        
        stats = {
            "success": True,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_stored": added,
            "message": f"Successfully ingested {len(documents)} documents into {len(chunks)} chunks"
        }
        
        print(f"\n{'='*50}")
        print(f"[OK] INGESTION COMPLETE")
        print(f"   Documents: {stats['documents_loaded']}")
        print(f"   Chunks: {stats['chunks_created']}")
        print(f"{'='*50}\n")
        
        return stats
    
    def ingest_uploaded_files(
        self,
        file_paths: List[str],
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest specific uploaded files.
        
        Args:
            file_paths: List of file paths to ingest
            clear_existing: Whether to clear existing documents first
            
        Returns:
            Ingestion statistics
        """
        if clear_existing:
            self.vector_store.clear()
        
        all_documents = []
        for file_path in file_paths:
            docs = load_documents(file_path)
            all_documents.extend(docs)
        
        if not all_documents:
            return {
                "success": False,
                "documents_loaded": 0,
                "chunks_created": 0,
                "message": "No valid documents found in uploaded files"
            }
        
        chunks = chunk_documents(all_documents)
        added = self.vector_store.add_documents(chunks)
        
        self._documents_ingested = True
        
        return {
            "success": True,
            "files_processed": len(file_paths),
            "documents_loaded": len(all_documents),
            "chunks_created": len(chunks),
            "chunks_stored": added,
            "message": f"Successfully ingested {len(file_paths)} files"
        }
    
    def query(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Response with answer and citations
        """
        print(f"\n{'='*50}")
        print(f"[*] QUERY: {question[:50]}...")
        print(f"{'='*50}")
        
        # Check if documents are ingested
        stats = self.vector_store.get_stats()
        if stats["document_count"] == 0:
            return {
                "answer": "[*] No documents have been ingested yet. Please upload and ingest documents first using the sidebar.",
                "citations": [],
                "retrieved_chunks": 0,
                "has_context": False
            }
        
        # Retrieve relevant chunks
        print("\n[*] Retrieving relevant chunks...")
        results = self.retriever.retrieve(question, top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information in the documents. Try rephrasing your question or ingesting more relevant documents.",
                "citations": [],
                "retrieved_chunks": 0,
                "has_context": False
            }
        
        # Format context
        context = self.retriever.retrieve_with_context(question, top_k)
        sources = self.retriever.get_sources(results)
        
        # Generate response
        print("\n[*] Generating response...")
        response = self.generator.generate(
            question=question,
            context=context,
            sources=sources
        )
        
        response["retrieved_chunks"] = len(results)
        
        print(f"\n{'='*50}")
        print(f"[OK] RESPONSE GENERATED")
        print(f"   Retrieved: {len(results)} chunks")
        print(f"   Citations: {len(sources)}")
        print(f"{'='*50}\n")
        
        return response
    
    def query_streaming(
        self,
        question: str,
        top_k: int = TOP_K
    ):
        """
        Process a query with streaming response.
        
        Yields:
            Response chunks and final metadata
        """
        # Check documents
        stats = self.vector_store.get_stats()
        if stats["document_count"] == 0:
            yield {
                "type": "error",
                "content": "No documents ingested yet"
            }
            return
        
        # Retrieve
        results = self.retriever.retrieve(question, top_k)
        if not results:
            yield {
                "type": "error", 
                "content": "No relevant information found"
            }
            return
        
        context = self.retriever.retrieve_with_context(question, top_k)
        sources = self.retriever.get_sources(results)
        
        # Yield sources first
        yield {
            "type": "sources",
            "content": sources
        }
        
        # Stream response
        for chunk in self.generator.generate_streaming(question, context):
            yield {
                "type": "content",
                "content": chunk
            }
        
        # Yield completion
        yield {
            "type": "done",
            "retrieved_chunks": len(results)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        vs_stats = self.vector_store.get_stats()
        return {
            "vector_store": vs_stats,
            "documents_ingested": self._documents_ingested or vs_stats["document_count"] > 0
        }
    
    def clear_all(self) -> bool:
        """Clear all ingested documents."""
        success = self.vector_store.clear()
        self._documents_ingested = False
        return success


# Singleton instance
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get the singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


if __name__ == "__main__":
    # Test pipeline
    pipeline = RAGPipeline()
    
    # Ingest sample documents
    from config import SAMPLE_DIR
    stats = pipeline.ingest_documents(str(SAMPLE_DIR))
    print(f"\nIngestion stats: {stats}")
    
    # Test query
    if stats["success"]:
        result = pipeline.query("What are the main topics in these documents?")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nCitations: {result['citations']}")
