"""
RAG Chatbot Configuration
Centralized settings for reproducibility
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
SAMPLE_DIR = DATA_DIR / "sample"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Split hierarchy

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# Using sentence-transformers (runs locally on CPU)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight model
# Alternative: "all-mpnet-base-v2" for better quality

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1)

# =============================================================================
# LLM SETTINGS
# =============================================================================
# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Google Settings (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-flash-latest")

TEMPERATURE = 0.1  # Low temperature for factual responses
MAX_TOKENS = 1024

# Model Provider priority: Google (if key exists) > OpenAI (if key exists) > Local
if GOOGLE_API_KEY:
    DEFAULT_LLM_PROVIDER = "google"
elif OPENAI_API_KEY:
    DEFAULT_LLM_PROVIDER = "openai"
else:
    DEFAULT_LLM_PROVIDER = "local"

# Local LLM (optional)
# Set USE_LOCAL_LLM=true in .env to force local use
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# =============================================================================
# VECTOR STORE SETTINGS
# =============================================================================
COLLECTION_NAME = "rag_documents"

# =============================================================================
# UI SETTINGS
# =============================================================================
APP_TITLE = "RAG Chatbot"
APP_DESCRIPTION = "Ask questions about your documents and get citation-backed answers!"
