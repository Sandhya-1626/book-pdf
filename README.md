# Aero-GPT: Antigravity & Theoretical Propulsion Research Portal

Aero-GPT is a specialized RAG (Retrieval-Augmented Generation) application designed for analyzing academic papers on propulsion and physics.

## 🚀 How to Run (3 Simple Steps)

### Step 1: Install Ollama & Pull Llama3
Aero-GPT uses a local LLM for maximum privacy and performance.
1. Download and install **Ollama** from [ollama.com](https://ollama.com/).
2. Run the following command in your terminal:
   ```bash
   ollama pull llama3
   ```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Aero-GPT
```bash
streamlit run app.py
```

---

## Technical Stack
- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector DB**: FAISS (In-memory)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Local LLM**: Ollama (Llama 3)