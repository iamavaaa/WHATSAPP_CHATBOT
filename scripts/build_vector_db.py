import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
SAMPLE_FILE = "../data/sample_common_crawl.jsonl"
CHROMA_DB_DIR = "../data/chroma_db"

def build_vector_db():
    print("Loading data from JSONL...")
    
    # We will read the JSONL file and extract the "text" field
    documents = []
    with open(SAMPLE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                documents.append(data['text'])
            except json.JSONDecodeError:
                continue
            
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
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"Success! Vector database saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    build_vector_db()
