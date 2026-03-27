# src/ingest/chunk_docs.py
from pathlib import Path
import json

RAW_DIR = Path("data/raw_docs")
OUT_DIR = Path("data/processed_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1200
OVERLAP = 200

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap

def main():
    for file in RAW_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks = []
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "id": f"{file.stem}_{i}",
                "source_file": file.name,
                "text": chunk
            })
        out_file = OUT_DIR / f"{file.stem}.json"
        out_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        print(f"Wrote {out_file.name} with {len(chunks)} chunks")

if __name__ == "__main__":
    main()