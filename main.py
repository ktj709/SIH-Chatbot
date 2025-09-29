"""
Main script that can be used for quick CLI indexing + query demo.
It wires all modules together and offers a minimal CLI.
"""
import argparse
from utils.load_pdf import load_pdf
from utils.fetch_wikipedia import fetch_wikipedia_page
from utils.fetch_plain_text import fetch_plain_text_url
from chunking import chunk_documents
from embed_store import EmbedStore
from retrieval import Retriever
from generate_answer import AnswerGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_pdf", help="Path to PDF to index", default=None)
    parser.add_argument("--index_wiki", help="Wikipedia page title to index (optional)", default=None)
    parser.add_argument("--index_url", help="Plain text URL to index", default=None)
    parser.add_argument("--question", help="Ask a question after indexing", default=None)
    args = parser.parse_args()

    store = EmbedStore()
    docs = []
    if args.index_pdf:
        docs += load_pdf(args.index_pdf)
    if args.index_wiki:
        docs.append(fetch_wikipedia_page(args.index_wiki))
    if args.index_url:
        docs.append(fetch_plain_text_url(args.index_url))

    if docs:
        chunks = chunk_documents(docs)
        print(f"Indexing {len(chunks)} chunks...")
        store.build_index(chunks)
        print("Index built.")
    else:
        print("No docs provided for indexing. Using existing index if present.")

    if args.question:
        retriever = Retriever(store)
        hits = retriever.retrieve_top_chunks(args.question, top_k=5)
        
        generator = AnswerGenerator()
        ans = generator.generate(args.question, hits)
        
        print("ANSWER:")
        print(ans)
        print("\nTop chunks:")
        for h in hits:
            print(h["id"], h["metadata"], h["text"][:200].replace("\n"," ") + "...")

if __name__ == "__main__":
    main()