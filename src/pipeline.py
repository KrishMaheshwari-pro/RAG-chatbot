"""
RAG Pipeline - Core document processing and query system
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
    EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD,
    OPENAI_API_KEY, OPENAI_MODEL,
    GOOGLE_API_KEY, GOOGLE_MODEL,
    TEMPERATURE, MAX_TOKENS,
    DEFAULT_LLM_PROVIDER, USE_LOCAL_LLM, OLLAMA_MODEL,
    CHROMA_DIR, COLLECTION_NAME
)


class RAGPipeline:
    """Main RAG Pipeline for document ingestion and querying"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        self.llm = self._initialize_llm()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )
        
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if USE_LOCAL_LLM or DEFAULT_LLM_PROVIDER == "local":
            try:
                from langchain_community.llms import Ollama
                return Ollama(model=OLLAMA_MODEL)
            except ImportError:
                print("[WARNING] Ollama not available, falling back to Google Gemini")
        
        if DEFAULT_LLM_PROVIDER == "google" and GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(
                model=GOOGLE_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS
            )
        elif OPENAI_API_KEY:
            return ChatOpenAI(
                model_name=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        else:
            raise ValueError(
                "[ERROR] No LLM provider configured. "
                "Set OPENAI_API_KEY or GOOGLE_API_KEY in .env"
            )
    
    def _load_document(self, file_path: str):
        """Load a single document based on file type"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif suffix in [".pptx", ".ppt"]:
                loader = UnstructuredPowerPointLoader(str(file_path))
            elif suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                return None
            
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"[WARNING] Error loading {file_path}: {str(e)}")
            return None
    
    def ingest_documents(self, directory: str, clear_existing: bool = False) -> Dict[str, Any]:
        """Ingest documents from a directory"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return {
                    "success": False,
                    "message": f"Directory not found: {directory}"
                }
            
            # Clear if requested
            if clear_existing:
                self._clear_vectorstore()
            
            # Find all supported files
            supported_extensions = [".pdf", ".pptx", ".ppt", ".md", ".txt"]
            files = []
            for ext in supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
                files.extend(directory.glob(f"**/*{ext}"))  # Recursive
            
            if not files:
                return {
                    "success": False,
                    "message": f"No supported documents found in {directory}"
                }
            
            # Load and process documents
            all_docs = []
            for file_path in files:
                docs = self._load_document(str(file_path))
                if docs:
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source"] = file_path.name
                    all_docs.extend(docs)
            
            if not all_docs:
                return {
                    "success": False,
                    "message": "No documents could be loaded"
                }
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=SEPARATORS
            )
            chunks = splitter.split_documents(all_docs)
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
            return {
                "success": True,
                "message": f"Ingested {len(files)} document(s) into {len(chunks)} chunks"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error ingesting documents: {str(e)}"
            }
    
    def ingest_uploaded_files(self, file_paths: List[str], clear_existing: bool = False) -> Dict[str, Any]:
        """Ingest uploaded files"""
        try:
            if clear_existing:
                self._clear_vectorstore()
            
            # Load and process documents
            all_docs = []
            for file_path in file_paths:
                docs = self._load_document(file_path)
                if docs:
                    file_name = Path(file_path).name
                    for doc in docs:
                        doc.metadata["source"] = file_name
                    all_docs.extend(docs)
            
            if not all_docs:
                return {
                    "success": False,
                    "message": "No documents could be loaded"
                }
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=SEPARATORS
            )
            chunks = splitter.split_documents(all_docs)
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
            return {
                "success": True,
                "message": f"Ingested {len(file_paths)} file(s) into {len(chunks)} chunks"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error ingesting files: {str(e)}"
            }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search_with_score(
                question,
                k=TOP_K
            )
            
            if not docs:
                return {
                    "answer": "[INFO] No relevant documents found.",
                    "citations": []
                }
            
            # Filter by similarity threshold
            relevant_docs = [
                (doc, score) for doc, score in docs
                if score >= SIMILARITY_THRESHOLD
            ]
            
            if not relevant_docs:
                return {
                    "answer": "[INFO] No documents met the similarity threshold.",
                    "citations": []
                }
            
            # Create context from retrieved documents
            context = "\n\n".join([
                f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc, score in relevant_docs
            ])
            
            # Create prompt
            prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Generate answer
            response = self.llm.invoke(
                prompt.format(context=context, question=question)
            )
            
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            # Format citations
            citations = []
            for idx, (doc, score) in enumerate(relevant_docs, 1):
                citations.append({
                    "citation": idx,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None),
                    "similarity": float(score)
                })
            
            return {
                "answer": answer_text,
                "citations": citations
            }
            
        except Exception as e:
            return {
                "answer": f"[ERROR] Error processing query: {str(e)}",
                "citations": []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            # Get collection info
            collection = self.vectorstore._collection
            doc_count = collection.count()
            
            return {
                "vector_store": {
                    "document_count": doc_count,
                    "collection_name": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL
                },
                "retrieval": {
                    "top_k": TOP_K,
                    "similarity_threshold": SIMILARITY_THRESHOLD
                },
                "llm": {
                    "provider": DEFAULT_LLM_PROVIDER,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS
                }
            }
        except Exception as e:
            return {
                "vector_store": {
                    "document_count": 0,
                    "collection_name": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL
                },
                "error": str(e)
            }
    
    def clear_all(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            self._clear_vectorstore()
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing vectorstore: {str(e)}")
            return False
    
    def _clear_vectorstore(self):
        """Clear the vector store"""
        try:
            # Delete and recreate collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR)
            )
        except Exception as e:
            print(f"[WARNING] Error clearing vectorstore: {str(e)}")
            # Try alternative method
            try:
                self.vectorstore._collection.delete(where={})
            except:
                pass


# Global pipeline instance
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
