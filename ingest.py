import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"


def load_documents():
    docs = []
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())
    return docs


def main():
    print("Loading documents from data/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print("No documents found. Add PDFs or .txt files to the data/ folder first.")
        return

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings (free, local model — first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    print(f"Done. Vector store saved to ./{DB_DIR}")


if __name__ == "__main__":
    main()