import os
import json
import mmh3  # Fast hashing library
from datasets import load_dataset
from tqdm import tqdm

# Configuration
# The interview requirement is 2-3GB. We'll set the target to 2.5GB.
TARGET_SIZE_BYTES = 2.5 * 1024 * 1024 * 1024  # 2.5 GB
# For the actual RAG demo, we will use a smaller sample (e.g., 50MB) to save time/costs on embeddings.
SAMPLE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

OUTPUT_FILE = "../data/cleaned_common_crawl.jsonl"
SAMPLE_FILE = "../data/sample_common_crawl.jsonl"

def clean_text(text):
    """Basic cleaning: remove excessive whitespace and extremely short lines."""
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 30]
    return ' '.join(cleaned_lines)

def is_quality_content(text):
    """Filter out garbage data."""
    if len(text) < 200: return False # Too short to be useful context
    if text.count('{') > 5 or text.count('<html') > 0: return False # Likely code or raw HTML
    return True

def build_dataset():
    # We use a set to store hashes of documents we've already seen (Deduplication)
    seen_hashes = set()
    current_size = 0
    sample_size = 0
    
    # Load a Common Crawl based dataset in streaming mode (doesn't download the whole thing at once)
    print("Connecting to Common Crawl dataset stream (C4)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_full, \
         open(SAMPLE_FILE, 'w', encoding='utf-8') as f_sample:
         
        with tqdm(total=TARGET_SIZE_BYTES, unit='B', unit_scale=True, desc="Building Dataset") as pbar:
            for row in dataset:
                raw_text = row['text']
                
                # 1. Clean the text
                cleaned = clean_text(raw_text)
                
                # 2. Quality Check
                if not is_quality_content(cleaned):
                    continue
                    
                # 3. Deduplication (Check if we've seen this exact text before)
                # mmh3 is much faster and uses less memory than standard hashlib
                doc_hash = mmh3.hash128(cleaned)
                if doc_hash in seen_hashes:
                    continue # Skip redundant data
                    
                # If it's new, add it to our seen list
                seen_hashes.add(doc_hash)
                
                # 4. Save to files
                data_to_write = json.dumps({"text": cleaned}) + "\n"
                bytes_written = len(data_to_write.encode('utf-8'))
                
                # Write to the full 2.5GB file
                f_full.write(data_to_write)
                current_size += bytes_written
                pbar.update(bytes_written)
                
                # Write to the sample file if we haven't reached the sample limit
                if sample_size < SAMPLE_SIZE_BYTES:
                    f_sample.write(data_to_write)
                    sample_size += bytes_written
                
                # Stop if we hit our 2-3GB target
                if current_size >= TARGET_SIZE_BYTES:
                    print(f"\nTarget size reached! Dataset saved to {OUTPUT_FILE}")
                    print(f"A smaller sample for the RAG demo was saved to {SAMPLE_FILE}")
                    break

if __name__ == "__main__":
    # Note: Running this script to 2.5GB will take some time and network bandwidth.
    build_dataset()
