"""
Fetch a Wikipedia page's plain text (summary + sections) via the public API.
"""
import requests
from typing import Dict

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/mobile-sections/"

def fetch_wikipedia_page(title: str) -> Dict:
    """
    Returns a dict with source, url, and extracted text.
    """
    title = title.replace(" ", "_")
    url = f"{WIKI_API}{title}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    # mobile-sections returns 'lead' and 'remaining' HTML; we'll extract plain text crudely
    texts = []
    if 'lead' in data and 'sections' in data['lead']:
        for sec in data['lead']['sections']:
            if 'text' in sec:
                texts.append(sec['text'])
    if 'remaining' in data and 'sections' in data['remaining']:
        for sec in data['remaining']['sections']:
            if 'text' in sec:
                texts.append(sec['text'])
    # remove HTML tags crudely
    from bs4 import BeautifulSoup
    plain = "\n\n".join(BeautifulSoup(t, "html.parser").get_text(separator="\n") for t in texts)
    return {
        "source": f"wikipedia:{title}",
        "url": f"https://en.wikipedia.org/wiki/{title}",
        "text": plain
    }
