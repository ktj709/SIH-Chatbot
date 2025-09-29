"""
LLM client wrapper.
Supports:
 - Google Generative (Gemini) if `google.generativeai` is installed and GEMINI_API_KEY present
 - Fallback to OpenAI's chat completions if OPENAI_API_KEY present

It exposes generate_answer(query, context_chunks, max_tokens).
"""
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Try import google generative
_generative_available = False
try:
    import google.generativeai as genai
    _generative_available = True
except Exception:
    _generative_available = False

try:
    import openai
    openai.api_key = OPENAI_API_KEY
    _openai_available = True if OPENAI_API_KEY else False
except Exception:
    _openai_available = False

def _call_gemini(prompt: str, max_tokens: int = 512):
    if not _generative_available:
        raise RuntimeError("google.generativeai not installed")
    genai.configure(api_key=GEMINI_API_KEY)
    # using simple text generation call (API details may vary with google SDK)
    response = genai.generate_text(model="gemini-2.0-flash", input=prompt, max_output_tokens=max_tokens)
    # SDK returns .text or similar
    return getattr(response, "text", str(response))

def _call_openai(prompt: str, max_tokens: int = 512):
    if not _openai_available:
        raise RuntimeError("OpenAI not configured")
    resp = openai.ChatCompletion.create(
        model="gpt-4o" if False else "gpt-4-0613",  # change per your access
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"]

def generate_answer(query: str, context_chunks: list, max_tokens: int = 512) -> str:
    """
    Combine query and context chunks into a system prompt and call the configured LLM.
    Each chunk should include metadata 'metadata' with source/page where available.
    """
    # build context string with citations
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        header = f"[{i+1}] source={source} page={page}"
        body = chunk.get("text", "")
        context_lines.append(f"{header}\n{body}\n")
    context_str = "\n\n".join(context_lines)

    system = (
        "You are a helpful tutor/assistant with access to educational slide content. "
        "Use the provided context chunks to answer the user question. Cite chunk indices "
        "like [1], [2] with source and page. If the answer is not in the material, say so and provide a short external explanation."
    )
    prompt = f"{system}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer (concise, cite chunks):"

    # Prefer Gemini if available
    if GEMINI_API_KEY and _generative_available:
        try:
            return _call_gemini(prompt, max_tokens=max_tokens)
        except Exception as e:
            print("Gemini call failed, falling back to OpenAI:", e)
    if OPENAI_API_KEY and _openai_available:
        return _call_openai(prompt, max_tokens=max_tokens)
    # last resort: a local simple heuristic answer
    # If no LLM configured, return stitched context snippet and basic reply
    reply = "No cloud LLM configured. Here are the most relevant context snippets:\n\n" + "\n\n".join(
        [f"[{i+1}] {c['text'][:400]}..." for i,c in enumerate(context_chunks[:3])]
    )
    return reply
