import os
import glob
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# LangChain text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Community (PDF loader, Chroma)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# HuggingFace embeddings (free, local, no API key needed)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# LangChain chain
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# ---------- Configuration ----------
PDF_FOLDER = "data"                     # Folder containing your PDFs
PERSIST_DIR = "chroma_db"               # Where Chroma saves the vector index

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Sentence-Transformers embedding model (local, no API key needed)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Global objects (initialized on startup)
vectorstore = None
qa_chain = None

# ---------- Helper Functions ----------
def get_embeddings():
    """Return local HuggingFace embeddings using sentence-transformers."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_llm():
    """Return the Groq LLM."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env")
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
        max_retries=2
    )

def load_pdf_documents():
    """Load all PDFs from PDF_FOLDER and return documents."""
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in folder '{PDF_FOLDER}'")

    all_docs = []
    for pdf_path in pdf_files:
        app.logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        all_docs.extend(loader.load())
    return all_docs

def index_documents():
    """Load all PDFs, chunk, embed with HuggingFace, and store in Chroma."""
    global vectorstore
    app.logger.info("Starting indexing process...")

    docs = load_pdf_documents()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    app.logger.info(f"Split document(s) into {len(chunks)} chunks")

    # Generate embeddings and store (overwrites existing index with all current PDFs)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    app.logger.info("Indexing complete. Vectors saved to disk.")

def rebuild_qa_chain():
    """(Re)build the QA chain using the current vectorstore."""
    global qa_chain
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    app.logger.info("QA chain is ready.")

def load_or_index():
    """Load existing vectorstore if present, otherwise index from scratch."""
    global vectorstore

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        app.logger.info("Loading existing vectorstore from disk...")
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        index_documents()

    rebuild_qa_chain()

# ---------- Flask Routes ----------
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files (CSS, etc.)."""
    return send_from_directory("static", filename)

@app.route("/ask", methods=["POST"])
def ask():
    """Handle user question, get answer from Groq, and return sources."""
    global qa_chain
    if not qa_chain:
        return jsonify({"error": "System not ready. Check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Expected JSON."}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result["source_documents"]

        sources = []
        for doc in source_docs:
            page = doc.metadata.get("page", "Unknown")
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources.append({
                "page": page,
                "snippet": snippet
            })

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        app.logger.exception("Error while processing question")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """Accept a new PDF, save it to data/, and re-index everything."""
    global vectorstore, qa_chain

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    os.makedirs(PDF_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(PDF_FOLDER, filename)
    file.save(filepath)

    try:
        app.logger.info(f"Re-indexing after upload: {filename}")
        index_documents()
        rebuild_qa_chain()
        return jsonify({"message": f"'{filename}' uploaded and indexed successfully."})
    except Exception as e:
        app.logger.exception("Error while indexing uploaded file")
        return jsonify({"error": str(e)}), 500

# ---------- Start the App ----------
if __name__ == "__main__":
    load_or_index()
    app.run(debug=True, host="0.0.0.0", port=5000)