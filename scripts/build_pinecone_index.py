import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


load_dotenv()

from rag_jsonl import load_texts_from_jsonl_files, rag_data_jsonl_paths

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "whatsapp-rag-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")


def ensure_index(pc: Pinecone) -> None:
    raw = pc.list_indexes()
    existing_names: set[str] = set()

    # New SDK usually returns an object that supports .names().
    if hasattr(raw, "names"):
        existing_names = set(raw.names())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                existing_names.add(item)
            elif isinstance(item, dict) and item.get("name"):
                existing_names.add(str(item["name"]))
            elif hasattr(item, "name"):
                existing_names.add(str(item.name))
            else:
                existing_names.add(str(item))
    else:
        # Fallback for non-list response shapes.
        existing_names.add(str(raw))

    if INDEX_NAME not in existing_names:
        raise RuntimeError(
            "Pinecone index not found. Create a SERVERLESS index first in Pinecone console "
            f"with name='{INDEX_NAME}', dimension=384 (for all-MiniLM-L6-v2), metric='cosine', then rerun this script. "
            f"Visible indexes={sorted(existing_names)}"
        )


def main() -> None:
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is required to build Pinecone index.")

    sample_paths = rag_data_jsonl_paths(BASE_DIR)
    missing = [p for p in sample_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "RAG data file(s) not found: "
            + ", ".join(str(p) for p in missing)
            + ". Set RAG_DATA_JSONL (comma-separated) or run crawl / prepare scripts first."
        )

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index(pc)
    index = pc.Index(INDEX_NAME)

    print("Loading documents from:", ", ".join(str(p) for p in sample_paths))
    docs = load_texts_from_jsonl_files(sample_paths)
    print(f"Loaded {len(docs)} records")

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunked_docs = splitter.create_documents(docs)
    print(f"Created {len(chunked_docs)} chunks")

    print("Uploading embeddings to Pinecone...")
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2"))

    # Upsert in batches to avoid huge single requests.
    batch_size = 100
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i : i + batch_size]
        texts = [d.page_content for d in batch]
        vectors = embeddings.embed_documents(texts)
        payload = []
        for j, (text, vec) in enumerate(zip(texts, vectors)):
            payload.append(
                {
                    "id": f"doc-{i + j}",
                    "values": vec,
                    "metadata": {"text": text},
                }
            )
        index.upsert(vectors=payload, namespace=PINECONE_NAMESPACE)
        if i % 1000 == 0:
            print(f"Upserted {min(i + batch_size, len(chunked_docs))}/{len(chunked_docs)}")

    print(f"Done. Index populated: {INDEX_NAME}")


if __name__ == "__main__":
    main()
