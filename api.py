"""
Simple FastAPI wrapper exposing endpoints to:
 - upload pdf (index)
 - query the indexed data
This demonstrates programmatic usage; you can run with: uvicorn api:app --reload
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
import os
from utils.load_pdf import load_pdf
from chunking import chunk_documents
from embed_store import EmbedStore
from retrieval import Retriever
from generate_answer import AnswerGenerator

app = FastAPI()
storage_dir = "uploaded_pdfs"
os.makedirs(storage_dir, exist_ok=True)

embed_store = EmbedStore()
retriever = Retriever(embed_store)
generator = AnswerGenerator()

class QueryIn(BaseModel):
    question: str
    top_k: int = 5

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    filename = os.path.join(storage_dir, file.filename)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    docs = load_pdf(filename)
    chunks = chunk_documents(docs)
    embed_store.build_index(chunks)
    return {"status":"indexed", "file": file.filename, "chunks_indexed": len(chunks)}

@app.post("/query/")
async def answer_query(q: QueryIn):
    hits = retriever.retrieve_top_chunks(q.question, top_k=q.top_k)
    answer = generator.generate(q.question, hits)
    return {"answer": answer, "hits": hits}