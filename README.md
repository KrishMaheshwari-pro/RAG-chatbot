# [DOCS] RAG Chatbot

A Domain-Specific Retrieval-Augmented Generation (RAG) chatbot for document Q&A with citation-backed answers.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)

## [FEATURE] Features

- **[FILE] Multi-format Document Support**: PDF, PowerPoint, Markdown, Text files
- **[SEARCH] Semantic Search**: Find relevant information using AI embeddings
- **[CHAT] Citation-backed Answers**: Every response includes source references
- **  Modern UI**: Beautiful dark-themed chat interface
- **[SAVE] Persistent Storage**: Documents survive restarts with ChromaDB
- **  Configurable**: Chunk size, top-k, model selection, and more

## [ARCH] Architecture

```
             
   Documents      Chunker       Embeddings  
  (PDF/PPTX)          (500 chars)          (MiniLM-L6)  
             
                                                 
                                                 
             
    Answer          LLM          ChromaDB   
  + Citations         (GPT-3.5)            (Vector DB)  
             
```

## [RUN] Quick Start

### 1. Clone & Setup

```bash
cd "d:\CODING\RAG Chatbot"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy example env file
copy .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## [DIR] Project Structure

```
RAG Chatbot/
  app.py                 # Streamlit main application
  config.py              # Configuration settings
  requirements.txt       # Dependencies
  .env.example          # Environment template
  data/
      documents/        # Your documents go here
      sample/           # Sample test documents
  src/
      document_loader.py # PDF/PPTX/MD loading
      chunker.py        # Text splitting
      embeddings.py     # Vector generation
      vector_store.py   # ChromaDB operations
      retriever.py      # Semantic search
      generator.py      # LLM responses
      pipeline.py       # RAG orchestration
  evaluation/
      metrics.py        # Evaluation metrics
  chroma_db/            # Vector database (auto-created)
```

##   Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `TOP_K` | 5 | Retrieved chunks per query |
| `OPENAI_MODEL` | gpt-3.5-turbo | LLM for generation |

## [STATS] Evaluation

Run evaluation metrics:

```python
from evaluation.metrics import evaluate_retrieval, evaluate_answer_faithfulness

# Evaluate retrieval quality
metrics = evaluate_retrieval(
    query="What is machine learning?",
    retrieved_docs=results,
    relevant_sources=["ml_chapter1.pdf"]
)
```

## [CONFIG] Development

### Adding New Document Types

1. Create loader function in `src/document_loader.py`
2. Add to `LOADER_MAP` dictionary
3. Test with sample document

### Using Local LLM (Ollama)

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama2`
3. Set in `.env`:
   ```
   USE_LOCAL_LLM=true
   OLLAMA_MODEL=llama2
   ```

## [NOTE] License

MIT License - feel free to use for learning and projects!

## [THANKS] Acknowledgments

- [LangChain](https://langchain.com) - LLM framework
- [ChromaDB](https://trychroma.com) - Vector database
- [Streamlit](https://streamlit.io) - Web UI framework
- [Sentence-Transformers](https://sbert.net) - Embeddings
