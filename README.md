# South African Budget RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query and compare South African national budget documents (2023–2026) using LLMs and vector search.

---

## 🚀 Features

- Loads multiple budget PDFs (2023–2026)
- Semantic search using embeddings (Sentence Transformers)
- Vector database powered by ChromaDB
- LLM responses via Groq (LLaMA 3)
- Cross-year budget comparison capability
- CLI chatbot interface

---

## 🏗️ Tech Stack

- LangChain (LCEL modern API)
- ChromaDB (vector database)
- HuggingFace Embeddings
- Groq LLM (LLaMA 3)
- PyPDF
- Python Dotenv

---

## 📁 Project Structure


src/
chain.py # Main RAG pipeline
ingest.py # PDF loading & chunking
llm.py # Groq LLM setup
vectorstore.py # Embeddings + Chroma DB
data/ # Budget PDFs (2023–2026)
db/ # Vector database (auto-generated)


---

## ⚙️ Setup

### 1. Clone repository
```bash
git clone https://github.com/your-username/sa-budget-rag.git
cd sa-budget-rag
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Add environment variables

Create .env file:

GROQ_API_KEY=your_api_key_here
5. Add PDFs

Place budget PDFs inside:

data/
▶️ Run the chatbot
python -m src.chain
💬 Example queries
Compare education spending from 2023 to 2026
What changed in VAT policy over the years?
Which year had the highest infrastructure allocation?
Summarize SA budget trends
