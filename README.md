# WhatsApp RAG Chatbot

This repository contains the code for a Retrieval-Augmented Generation (RAG) AI Chatbot designed to be deployed on WhatsApp.

## Project Structure

- `data/`: Contains the cleaned and deduplicated datasets.
- `scripts/`: Contains data engineering scripts (e.g., downloading and cleaning Common Crawl data).
- `src/`: Contains the core application logic (RAG pipeline, API server, WhatsApp integration).

## Step 1: Data Preparation

The interview task requires processing 2-3GB of Common Crawl data while removing redundant information. 

To achieve this efficiently, we use a streaming approach with the Hugging Face `datasets` library and `mmh3` for fast hashing and deduplication.

### Running the Data Prep Script

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   cd scripts
   python prepare_dataset.py
   ```

**Note on Architecture:** The script will generate a full 2.5GB file (`cleaned_common_crawl.jsonl`) to satisfy the data engineering requirement. However, embedding 2.5GB of text for a local RAG demo is extremely resource-intensive and expensive. Therefore, the script also generates a 50MB `sample_common_crawl.jsonl` file. The Vector Database in this project is populated using this smaller sample to keep the demo fast, cost-effective, and easy for evaluators to run locally.

## Step 2: Build the Vector Database

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build local embeddings + ChromaDB:
   ```bash
   cd scripts
   python build_vector_db.py
   ```

This creates `data/chroma_db`, used by the chatbot retriever.

## Step 3: Run Flask + Twilio Webhook + Gemini RAG

1. Create `.env` from `.env.example`:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   CHROMA_DB_DIR=data/chroma_db
   PORT=5000
   ```

2. Start the backend from project root:
   ```bash
   python src/app.py
   ```

3. Expose local server using ngrok:
   ```bash
   ngrok http 5000
   ```

4. In Twilio WhatsApp Sandbox settings:
   - Set **When a message comes in** to:
     - `https://<your-ngrok-domain>/whatsapp`
   - Method: `POST`

## End-to-End Flow

User (WhatsApp) -> Twilio -> Flask (`/whatsapp`) -> RAG retrieval (ChromaDB) -> Gemini generation -> Flask -> Twilio -> User.

## Notes for Interview Evaluation

- Context retention is per user (`From` phone number) in memory.
- Retrieval uses local ChromaDB built from cleaned Common Crawl sample.
- Generation uses Gemini via `GOOGLE_API_KEY`.
- For production scale, replace in-memory history with Redis/Postgres and add queue/retry logic.

## Railway Deployment

1. Push this project to GitHub.
2. In Railway, click **New Project** -> **Deploy from GitHub repo**.
3. Add environment variables in Railway project settings:
   - `GOOGLE_API_KEY=your_gemini_api_key`
   - `GEMINI_MODEL=gemini-1.5-flash`
   - `CHROMA_DB_DIR=data/chroma_db`
4. Railway uses the `Procfile` start command automatically:
   - `gunicorn src.app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Once deployed, copy Railway public URL and set Twilio webhook to:
   - `https://<railway-domain>/whatsapp`
   - Method: `POST`

### Important

- If `data/chroma_db` is not present in deployment, retrieval will have no indexed context.
- For production, use a managed vector DB (Pinecone/Qdrant/pgvector) instead of local filesystem storage.

## Pinecone (Production-Grade Retrieval)

This project supports Pinecone as a managed vector database for deployment environments.

### 1) Create and Populate Pinecone Index

Set env vars locally in `.env`:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=whatsapp-rag-index
GOOGLE_API_KEY=your_gemini_api_key
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
```

Run:

```bash
cd scripts
python build_pinecone_index.py
```

If the script says the index is missing, create it once in Pinecone Console:

- Type: **Serverless**
- Name: `whatsapp-rag-index` (or your `PINECONE_INDEX_NAME`)
- Dimension: `1536`
- Metric: `cosine`
- Cloud/Region: any supported starter option

### 2) Enable Pinecone in App

In deployment env vars (Render/Railway):

```bash
USE_PINECONE=true
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=whatsapp-rag-index
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

With this setup, retrieval comes from Pinecone and no local `data/chroma_db` is required.
