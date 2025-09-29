"""
Answer generation module using Gemini 2.5 Flash.
Takes a query + retrieved context and returns a natural language answer.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


class AnswerGenerator:
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query: str, context_chunks: list) -> str:
        """
        query: user question
        context_chunks: list of dicts from retriever [{text, metadata, ...}]
        """
        context_text = "\n\n".join(
            [f"Source (p.{c['metadata'].get('page', '?')}): {c['text']}" for c in context_chunks]
        )

        prompt = f"""
You are an educational assistant. Answer the question using the provided context.
If relevant, cite the slide/page numbers from metadata.

Question:
{query}

Context:
{context_text}

Answer:
"""

        print("\n🔎 DEBUG: Prompt being sent to Gemini:\n", prompt)

        response = self.model.generate_content(prompt)
        return response.text.strip()