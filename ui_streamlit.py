"""
Minimal Streamlit UI to upload PDFs and ask questions.
Run: streamlit run ui_streamlit.py
"""
import streamlit as st
from api import embed_store, retriever  # import instances so we share same store
from utils.load_pdf import load_pdf
from chunking import chunk_documents
from generate_answer import AnswerGenerator

st.set_page_config(page_title="Slides Q&A Chatbot", layout="wide")

st.title("Slides Q&A Chatbot")

# Initialize the answer generator
@st.cache_resource
def get_generator():
    return AnswerGenerator()

generator = get_generator()

with st.sidebar:
    st.header("Index slides")
    uploaded = st.file_uploader("Upload slides PDF", type="pdf")
    if st.button("Index uploaded PDF") and uploaded is not None:
        # save temporarily
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        docs = load_pdf(temp_path)
        chunks = chunk_documents(docs)
        embed_store.build_index(chunks)
        st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

st.header("Ask a question")
question = st.text_input("Your question:")
top_k = st.slider("Top-k context chunks", 1, 10, 5)

if st.button("Ask") and question.strip():
    hits = retriever.retrieve_top_chunks(question, top_k=top_k)
    answer = generator.generate(question, hits)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Cited chunks")
    for i, h in enumerate(hits):
        meta = h.get("metadata", {})
        st.write(f"[{i+1}] id={h['id']} source={meta.get('source')} page={meta.get('page')} distance={h.get('distance')}")
        st.write(h['text'][:800] + ("..." if len(h['text'])>800 else ""))