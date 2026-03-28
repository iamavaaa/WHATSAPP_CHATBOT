# WhatsApp RAG Chatbot (COMMANDO Networks)

WhatsApp bot that answers questions about **COMMANDO Networks**—company info, website content, switches, wireless, routers, gateways, and accessories—using RAG over a **live crawl** of [commandonetworks.com](https://commandonetworks.com/) and optional curated snippets.

## Project Structure

- `data/`: JSONL corpora for RAG (`commando_networks.jsonl`, optional `commando_curated_facts.jsonl`); large raw crawl output is gitignored.
- `scripts/`: Crawl and dataset prep (`prepare_commando_dataset.py`, `crawl_commando_site.py`), Pinecone/Chroma builders.
- `src/`: Flask webhook, Twilio replies, RAG + Gemini.

## Step 1: Build the company dataset

Discovery uses **`robots.txt`** / **`sitemap.xml`** plus link crawling. Respect rate limits and site terms.

**Full pipeline** (large raw HTML corpus → cleaned JSONL sample, from repo root):

```bash
pip install -r requirements.txt
python scripts/prepare_commando_dataset.py
```

Outputs:

- `data/raw_commando_crawl.jsonl` — raw multi-pass HTML (assignment-scale size; mostly repeated pages).
- `data/commando_networks.jsonl` — merged text per URL, chunked, capped by `COMMANDO_SAMPLE_MB`.

**Rebuild sample only** (after you already have raw):

```bash
python scripts/prepare_commando_dataset.py --clean-only
```

**Quick smaller crawl** (fewer pages):

```bash
cd scripts
python crawl_commando_site.py
```

Optional: add **`data/commando_curated_facts.jsonl`** to `RAG_DATA_JSONL` (comma-separated) for short, retrieval-friendly facts aligned to the catalog.

Tune with env vars: `COMMANDO_TARGET_RAW_GB`, `COMMANDO_SAMPLE_MB`, `CRAWL_DELAY_SEC`, `COMMANDO_MAX_PASSES`, `COMMANDO_DISCOVERY_CAP`, `COMMANDO_SITEMAP_CAP`, chunking vars (see `prepare_commando_dataset.py` docstring).

## Step 2: Pinecone Vector Database

**Use Pinecone** for production (Render, Railway, etc.). Chroma under `data/chroma_db` is only for local dev if `USE_PINECONE=false`.

### Create and populate the index

In `.env` (repo root):

```bash
USE_PINECONE=true
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=whatsapp-rag-index-384
RAG_DATA_JSONL=data/commando_networks.jsonl,data/commando_curated_facts.jsonl
GOOGLE_API_KEY=your_gemini_api_key
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
```

Run:

```bash
cd scripts
python build_pinecone_index.py
```

`.env` is loaded from the **repository root** even when you run from `scripts/`.

Create the index once in Pinecone (if missing):

- Type: **Serverless**
- Name: matches `PINECONE_INDEX_NAME`
- Dimension: **384** (for `all-MiniLM-L6-v2`)
- Metric: **cosine**

## Step 3: Run Flask + Twilio + Gemini

1. From repo root:
   ```bash
   python src/app.py
   ```
2. Expose with ngrok: `ngrok http 5000`
3. Twilio WhatsApp sandbox → **When a message comes in** → `https://<your-ngrok-domain>/whatsapp` (POST)

## End-to-end flow

User (WhatsApp) → Twilio → Flask (`/whatsapp`) → Pinecone retrieval → Gemini → Twilio → User.

## Deployment env (e.g. Render/Railway)

```bash
USE_PINECONE=true
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=whatsapp-rag-index-384
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

Rebuild the Pinecone index whenever you change `RAG_DATA_JSONL` files.
