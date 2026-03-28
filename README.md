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

**Common Crawl vs the company website:** Common Crawl is a periodic snapshot of *many* sites. A specific domain like [commandonetworks.com](https://commandonetworks.com/) may appear in some crawls or not, and pages may be stale or incomplete. For a WhatsApp bot that must reflect **this company’s current site**, treat the **live crawl** (`prepare_commando_dataset.py` / `crawl_commando_site.py`) as the real product dataset—not C4.

### Company website corpus ([commandonetworks.com](https://commandonetworks.com/))

Discovery uses the **live** `robots.txt` `Sitemap:` entries and `sitemap.xml` (plus link BFS) so product and detail URLs are less likely to be missed.

To mirror the same pattern (large raw crawl → ~50MB cleaned JSONL for RAG), run from the **repository root**:

```bash
pip install -r requirements.txt
python scripts/prepare_commando_dataset.py
```

This writes `data/raw_commando_crawl.jsonl` (target ~2.5GB of HTML, built with multiple polite passes over the same URLs) and `data/commando_networks.jsonl` (one merged extract per URL, split into overlapping chunks, capped at `COMMANDO_SAMPLE_MB`). **Raw gigabytes are mostly repeated HTML; unique text from a single domain is often only a few MB**—that is normal. Rebuild the sample without re-crawling: `python scripts/prepare_commando_dataset.py --clean-only`. Set `RAG_DATA_JSONL=data/commando_networks.jsonl` for indexing.

Optional: add **`data/commando_curated_facts.jsonl`** to `RAG_DATA_JSONL` (comma-separated) only if you need guaranteed retrieval for facts that never appear as plain text on the site.

Tune `COMMANDO_TARGET_RAW_GB`, `COMMANDO_SAMPLE_MB`, `CRAWL_DELAY_SEC`, `COMMANDO_MAX_PASSES`, and `COMMANDO_SITEMAP_CAP` via environment variables if needed.

## Step 2: Pinecone Vector Database

**Use Pinecone** for real deployments (Render, Railway, etc.): the index stays in the cloud and survives redeploys. Chroma (`data/chroma_db`) is only a **local fallback** if you set `USE_PINECONE=false`.

### 1) Create and Populate Pinecone Index

Set env vars locally in `.env`:

```bash
USE_PINECONE=true
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=whatsapp-rag-index-384
RAG_DATA_JSONL=data/commando_networks.jsonl
GOOGLE_API_KEY=your_gemini_api_key
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
```

Set `RAG_DATA_JSONL` to your JSONL files (e.g. `data/commando_networks.jsonl,data/commando_curated_facts.jsonl`). If it points only at the small curated file, the index will have very few vectors.

Run:

```bash
cd scripts
python build_pinecone_index.py
```

(`.env` is loaded from the **repository root** even when you run this from `scripts/`.)

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
