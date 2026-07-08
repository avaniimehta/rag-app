"""
Loads raw text out of PDF, HTML, and Markdown files.
Each loader returns plain text - chunking happens separately.
"""
import os
from pathlib import Path


def load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def load_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # strip script/style/nav/footer - these aren't real content and can
    # otherwise produce huge garbage chunks (minified JS, tracking code)
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".md": load_md,
    ".txt": load_md,
}


def load_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext} (file: {path})")
    return LOADERS[ext](path)


def load_corpus(corpus_dir: str):
    """
    Yields (file_path, raw_text) for every supported file in corpus_dir.
    """
    for root, _, files in os.walk(corpus_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in LOADERS:
                full_path = os.path.join(root, fname)
                try:
                    text = load_file(full_path)
                    if text.strip():
                        yield full_path, text
                except Exception as e:
                    print(f"[loaders] WARNING: failed to load {full_path}: {e}")
