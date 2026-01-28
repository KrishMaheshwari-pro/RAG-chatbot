# RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red)](https://streamlit.io/) [![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange)](https://www.trychroma.com/)

A clean, production-ready Retrieval-Augmented Generation (RAG) chatbot for querying your documents with citation-backed answers.

---

## Key Features ✅

- Multi-format document ingestion: **PDF**, **PPTX**, **Markdown**, **TXT**
- Semantic search using sentence-transformer embeddings
- Chat UI with **citation-backed answers** and source listings
- Persistent vector storage (ChromaDB) — documents survive restarts
- Configurable chunking, retrieval (top-K), and LLM provider options (OpenAI / Google / local)
- Streamlit-based UI ready for local demos and prototyping

---

## Quick Start — Run locally (Windows)

1. Clone or copy the repository into your workspace.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Configure environment variables:

```powershell
copy .env.example .env
# Edit .env and set OPENAI_API_KEY or GOOGLE_API_KEY if using a cloud LLM provider
```

5. Start the Streamlit app:

```powershell
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
├─ app.py                 # Streamlit app
├─ config.py              # Application configuration (env-driven)
├─ requirements.txt       # Python dependencies
├─ data/                  # Document storage (add files to data/documents/)
├─ chroma_db/             # Persistent vector DB (auto-created)
├─ src/                   # Core pipeline and helpers
│  └─ pipeline.py         # Core ingestion, retrieval and query logic
└─ README.md
```

---

## Configuration & Environment Variables

Copy `.env.example` to `.env` and set any of the following as needed:

- OPENAI_API_KEY — OpenAI API key (optional)
- GOOGLE_API_KEY — Google Gemini API key (optional)
- USE_LOCAL_LLM — Set to `true` to use a local LLM (Ollama)
- CHUNK_SIZE, CHUNK_OVERLAP — Text splitting settings
- TOP_K — Number of retrieved chunks used to form context

See `config.py` for full defaults and notes.

---

## How it works (brief)

1. Documents are ingested and split into chunks (configurable chunk size/overlap).
2. Chunks are embedded with a sentence-transformer model and stored in ChromaDB.
3. Queries retrieve the most relevant chunks, build a context, and ask a configured LLM to answer with source citations.

---

## Development & Contributing

- Add new document loaders or LLM adapters inside `src/` and add tests.
- Keep changes small and focused — use one logical change per commit.
- Open issues for bugs or feature requests.

If you want me to help push this repository to GitHub and create incremental commits to grow your contribution graph, tell me and I will proceed.

---

## License

MIT License — see `LICENSE` (or add one) for full terms.

---

## Acknowledgements

Built with: LangChain, ChromaDB, Streamlit, and Sentence-Transformers.
