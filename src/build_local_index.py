# src/rag/build_local_index.py
from pathlib import Path
import json
import faiss
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import os

load_dotenv()

PROCESSED_DIR = Path("data/processed_docs")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
)
print(client)
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
print(EMBED_MODEL)

all_chunks = []
for file in PROCESSED_DIR.glob("*.json"):
    all_chunks.extend(json.loads(file.read_text(encoding="utf-8")))

texts = [c["text"] for c in all_chunks]

vectors = []
for i in range(0, len(texts), 32):
    batch = texts[i:i+32]
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    vectors.extend([d.embedding for d in resp.data])

arr = np.array(vectors, dtype="float32")
index = faiss.IndexFlatL2(arr.shape[1])
index.add(arr)

faiss.write_index(index, str(INDEX_DIR / "docs.faiss"))
(INDEX_DIR / "chunks.json").write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

print(f"Indexed {len(all_chunks)} chunks")