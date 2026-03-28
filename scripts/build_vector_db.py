import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

from rag_jsonl import load_texts_from_jsonl_files, rag_data_jsonl_paths

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"


def build_vector_db():
    print("Loading data from JSONL...")
    sample_paths = rag_data_jsonl_paths(BASE_DIR)
    missing = [p for p in sample_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "RAG data file(s) not found: "
            + ", ".join(str(p) for p in missing)
            + ". Set RAG_DATA_JSONL (comma-separated) or run crawl / prepare scripts first."
        )

    print("Files:", ", ".join(str(p) for p in sample_paths))
    documents = load_texts_from_jsonl_files(sample_paths)
    print(f"Loaded {len(documents)} documents.")
    
    # Text Splitting
    # We split the text into smaller chunks so the LLM can digest them easily
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Convert strings to LangChain Document objects
    texts = text_splitter.create_documents(documents)
    print(f"Created {len(texts)} chunks.")
    
    # Create Vector Store with local embeddings
    print("Generating embeddings and building ChromaDB...")
    print("This might take a few minutes depending on your CPU speed.")
    
    # Using Local HuggingFace Embeddings instead of Google API
    # This is FREE, runs locally, and avoids API rate limits!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We build and save the database locally to the CHROMA_DB_DIR
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    
    print(f"Success! Vector database saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    build_vector_db()
