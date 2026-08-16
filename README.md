# 📄 DocMind — RAG-Powered Document Chat

DocMind is a Retrieval-Augmented Generation (RAG) chat application that lets you upload PDF documents and ask natural-language questions about their content. It retrieves the most relevant passages from your documents and generates grounded, cited answers — instead of relying on an LLM's general knowledge alone.

Built entirely with a **free, open-source stack** — no paid API keys required.

---

## ✨ Features

- 🔍 **Retrieval-Augmented Generation** — answers are grounded in your actual documents, not hallucinated
- 📚 **Source citations** — every answer shows which page(s) it came from
- 📎 **Drag-and-drop PDF upload** — upload new documents directly from the browser; the app re-indexes automatically
- 💬 **Multiple conversations** — a ChatGPT-style sidebar to manage separate chat threads
- 🔎 **Search past chats** — filter conversations by title or message content
- ⏱️ **Timestamps** — every message is timestamped
- ⬇️ **Export chat history** — download any conversation as a `.txt` file
- ⌨️ **Typing animation** — responses stream in progressively
- 📝 **Markdown rendering** — bullet points, bold text, etc. render properly in answers
- 🎨 **Custom animated UI** — dark glassmorphism theme with an animated gradient background
- 💰 **100% free stack** — no OpenAI billing required

---

## 🏗️ Architecture

PDF Upload → Text Chunking → Local Embeddings → ChromaDB (vector store)
↓
User Question → Embed Question → Similarity Search → Top-K Chunks
↓
Chunks + Question → Groq LLM → Answer + Sources


**Pipeline steps:**
1. **Ingestion** — PDFs are loaded and split into ~500-character overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
2. **Embedding** — Each chunk is converted into a vector using a free local HuggingFace model (`all-MiniLM-L6-v2`) — runs entirely on-device, no API cost
3. **Storage** — Vectors are persisted in **ChromaDB**, a local vector database
4. **Retrieval** — Incoming questions are embedded the same way; ChromaDB returns the top-3 most similar chunks via cosine similarity
5. **Generation** — Retrieved chunks + the question are sent to **Groq's LLaMA 3.1 8B Instant** model for a grounded, fast response
6. **Response** — Flask returns the answer along with source page numbers to the frontend

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| RAG Orchestration | LangChain |
| Embeddings | HuggingFace `sentence-transformers` (local, free) |
| Vector Store | ChromaDB |
| LLM | Groq API (LLaMA 3.1 8B Instant) |
| Frontend | HTML, CSS, vanilla JavaScript |
| PDF Parsing | PyPDF |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 (recommended — newer versions may have dependency issues)
- A free [Groq API key](https://console.groq.com/keys)

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/itsmesaheli2005/docmind.git
   cd docmind
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the project root:

GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.1-8b-instant


5. **Add a PDF**
   Place a PDF file into the `data/` folder (or use the in-app drag-and-drop upload once running).

6. **Run the app**
```bash
   python app.py
```
   Open `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

docmind/
├── app.py # Flask backend, RAG pipeline, routes
├── ingest.py # Standalone script to (re)build the vector store
├── requirements.txt
├── .env # API keys (not committed)
├── data/ # PDF documents live here
├── chroma_db/ # Auto-generated vector store (not committed)
├── templates/
│ └── index.html # Chat UI
└── static/
└── style.css # Dark glassmorphism theme


---

## 🔮 Future Improvements

- [ ] Deploy publicly (Render / Railway)
- [ ] Show which specific document a source came from (for multi-PDF setups)
- [ ] Relevance/confidence scores alongside sources
- [ ] PDF export of chat history (currently `.txt`)
- [ ] User authentication for multi-user document sets

---

## 📄 License

This project is open source and available under the MIT License.

---

