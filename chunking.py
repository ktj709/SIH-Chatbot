"""
Split documents into smaller chunks and keep metadata mapping to source/page.
Simple sentence-based chunking with token/char limit per chunk.
"""
from typing import List, Dict
import re

DEFAULT_CHUNK_SIZE = 800  # characters per chunk (adjustable)
DEFAULT_OVERLAP = 200

def chunk_documents(docs: List[Dict], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """
    docs: list of {source, page (optional), text}
    returns list of chunks: {id, source, page, text}
    """
    chunks = []
    chunk_id = 0
    for doc in docs:
        text = doc.get("text", "")
        source = doc.get("source")
        page = doc.get("page", None)
        # naive sliding window by characters but break on sentence boundaries
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        current = ""
        for sent in sentences:
            if not sent.strip(): continue
            if len(current) + len(sent) + 1 <= chunk_size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "source": source,
                        "page": page,
                        "text": current
                    })
                    chunk_id += 1
                # start new chunk with overlap from last chunk
                # we implement overlap by taking last N chars of current if exists
                if overlap > 0:
                    overlap_text = current[-overlap:] if current else ""
                    current = (overlap_text + " " + sent).strip()
                else:
                    current = sent.strip()
        if current:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "source": source,
                "page": page,
                "text": current
            })
            chunk_id += 1
    return chunks
