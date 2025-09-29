"""
Load text from PDF files and return documents with metadata.
Uses pdfminer.six for robust extraction.
"""
from pdfminer.high_level import extract_text
from typing import List, Dict
import os

def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF and split by pages (if possible).
    Returns list of dicts: { 'source': filename, 'page': page_no, 'text': text }
    """
    assert os.path.exists(file_path), f"PDF not found: {file_path}"
    # pdfminer extract_text returns whole text. For simplicity, we'll split by form feed \f which pdfminer uses between pages.
    raw = extract_text(file_path)
    pages = raw.split("\f")
    docs = []
    filename = os.path.basename(file_path)
    for i, page_text in enumerate(pages):
        text = page_text.strip()
        if not text:
            continue
        docs.append({
            "source": filename,
            "page": i + 1,
            "text": text
        })
    return docs
