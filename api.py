"""
API routes for REST endpoint integration (optional)
"""

from typing import Dict, Any
from pipeline import get_pipeline


class RAGChatbotAPI:
    """REST API wrapper for RAG chatbot"""
    
    def __init__(self):
        self.pipeline = get_pipeline()
    
    def ingest_documents(self, file_paths: list, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Ingest documents via API
        
        Args:
            file_paths: List of file paths to ingest
            clear_existing: Whether to clear existing documents
        
        Returns:
            Result dictionary with success status and message
        """
        return self.pipeline.ingest_uploaded_files(file_paths, clear_existing)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
        
        Returns:
            Dictionary with answer and citations
        """
        return self.pipeline.query(question)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.pipeline.get_stats()
    
    def clear_all(self) -> bool:
        """Clear all documents"""
        return self.pipeline.clear_all()


# Example usage for FastAPI integration
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
api = RAGChatbotAPI()


class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    file_paths: list
    clear_existing: bool = False


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = api.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    try:
        result = api.ingest_documents(request.file_paths, request.clear_existing)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats_endpoint():
    return api.get_stats()


@app.delete("/clear")
async def clear_endpoint():
    api.clear_all()
    return {"message": "All data cleared"}
"""
