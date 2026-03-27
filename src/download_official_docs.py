# src/ingest/download_official_docs.py
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup

SRC_FILE = Path("data/source_urls.txt")
OUT_DIR = Path("data/raw_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "untitled"
    main = soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text

def main():
    urls = [u.strip() for u in SRC_FILE.read_text(encoding="utf-8").splitlines() if u.strip()]
    for url in urls:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        title, text = extract_main_text(resp.text)
        filename = slugify(title) + ".txt"
        content = f"SOURCE_URL: {url}\nTITLE: {title}\n\n{text}\n"
        (OUT_DIR / filename).write_text(content, encoding="utf-8")
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()