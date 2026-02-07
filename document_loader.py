"""
Document Loader Module
Handles loading documents from various formats: PDF, PPTX, Markdown
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

# Conditional imports for document loaders
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
except ImportError:
    from langchain.document_loaders import UnstructuredPowerPointLoader

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
except ImportError:
    from langchain.document_loaders import UnstructuredMarkdownLoader


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return documents."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "pdf"
    return documents


def load_pptx(file_path: str) -> List[Document]:
    """Load a PowerPoint file and return documents."""
    loader = UnstructuredPowerPointLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "pptx"
    return documents


def load_markdown(file_path: str) -> List[Document]:
    """Load a Markdown file and return documents."""
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "markdown"
    return documents


def load_text(file_path: str) -> List[Document]:
    """Load a plain text file and return documents."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document(
        page_content=content,
        metadata={
            "source": os.path.basename(file_path),
            "file_type": "text"
        }
    )
    return [doc]


# File extension to loader mapping
LOADER_MAP = {
    ".pdf": load_pdf,
    ".pptx": load_pptx,
    ".ppt": load_pptx,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
}


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext not in LOADER_MAP:
        print(f"[WARNING] Unsupported file type: {ext} for file {path.name}")
        return []
    
    try:
        loader_func = LOADER_MAP[ext]
        documents = loader_func(str(path))
        print(f"[OK] Loaded: {path.name} ({len(documents)} page(s))")
        return documents
    except Exception as e:
        print(f"[ERROR] Error loading {path.name}: {e}")
        return []


def load_documents(directory: str, extensions: Optional[List[str]] = None) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to the directory containing documents
        extensions: Optional list of file extensions to filter (e.g., ['.pdf', '.md'])
        
    Returns:
        List of Document objects from all loaded files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return []
    
    if not dir_path.is_dir():
        # Single file
        return load_document(str(dir_path))
    
    all_documents = []
    supported_extensions = extensions or list(LOADER_MAP.keys())
    
    # Recursively find all files
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            docs = load_document(str(file_path))
            all_documents.extend(docs)
    
    print(f"\n[*] Total documents loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    # Test loading
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import SAMPLE_DIR
    
    docs = load_documents(str(SAMPLE_DIR))
    for doc in docs[:3]:
        print(f"\n--- {doc.metadata.get('source', 'Unknown')} ---")
        print(doc.page_content[:200] + "...")
