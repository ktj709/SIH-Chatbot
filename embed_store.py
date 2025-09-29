"""
Create embeddings and a vector index.
This implementation uses sentence-transformers for embeddings
and chromadb as a vector store (PersistentClient).
"""

import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class EmbedStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = "chromadb_store"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.persist_dir = persist_dir

        # ✅ New API: PersistentClient (auto-saves to disk)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # single collection named 'slides'
        self.collection = self.client.get_or_create_collection(name="slides")

    def build_index(self, chunks: List[Dict], batch_size: int = 64):
        """
        Build or update the vector index.

        chunks: list of dicts with keys {'id', 'text', 'source', 'page'}
        """
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metas = [{"source": c.get("source"), "page": c.get("page")} for c in chunks]

        # compute embeddings in batches
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size
        ).tolist()

        # upsert into chroma (auto-persistent)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings
        )
        return True

    def query(self, query_text: str, top_k: int = 5):
        """
        Query the vector store with a text string.
        Returns top_k most similar chunks with id, text, metadata, and distance.
        """
        emb = self.model.encode([query_text])[0].tolist()

        results = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                hits.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
        return hits
