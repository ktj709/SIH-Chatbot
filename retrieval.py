"""
High-level retrieval interface that calls EmbedStore.query and formats top chunks.
"""

from typing import List, Dict
from embed_store import EmbedStore


class Retriever:
    def __init__(self, embed_store: EmbedStore):
        self.store = embed_store

    def retrieve_top_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        hits = self.store.query(query, top_k=top_k)
        return hits
