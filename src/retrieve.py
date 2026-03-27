
from pathlib import Path
import json
import faiss
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "index" / "docs.faiss"
CHUNKS_PATH = BASE_DIR / "data" / "index" / "chunks.json"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

index = faiss.read_index(str(INDEX_PATH))
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

def retrieve(query: str, top_k: int = 5):
    embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    ).data[0].embedding

    query_vector = np.array([embedding], dtype="float32")
    distances, ids = index.search(query_vector, top_k)

    results = []
    for idx in ids[0]:
        if idx >= 0:
            results.append(chunks[idx])

    return results