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

## Step 2: Pinecone Vector Database

This project supports Pinecone as a managed vector database for deployment environments.

### 1) Create and Populate Pinecone Index

Set env vars locally in `.env`:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=whatsapp-rag-index-384
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
- Name: `whatsapp-rag-index-384` (or your `PINECONE_INDEX_NAME`)
- Dimension: `384`
- Metric: `cosine`
- Cloud/Region: any supported starter option

## Step 3: Run Flask + Twilio Webhook + Gemini RAG

1. Start backend from project root:
   ```bash
   python src/app.py
   ```

2. Expose local server using ngrok:
   ```bash
   ngrok http 5000
   ```

3. In Twilio WhatsApp Sandbox settings:
   - Set **When a message comes in** to:
     - `https://<your-ngrok-domain>/whatsapp`
   - Method: `POST`

## End-to-End Flow

User (WhatsApp) -> Twilio -> Flask (`/whatsapp`) -> RAG retrieval (Pinecone) -> Gemini generation -> Flask -> Twilio -> User.

## App Configuration

In deployment env vars (Render/Railway):

```bash
USE_PINECONE=true
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=whatsapp-rag-index-384
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

With this setup, retrieval comes from Pinecone only.
