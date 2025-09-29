"""
Fetch plain text from an arbitrary URL (assumes HTML page).
Useful to fetch open educational resources that are public HTML.
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict

def fetch_plain_text_url(url: str) -> Dict:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    plain = "\n".join(lines)
    return {
        "source": url,
        "url": url,
        "text": plain
    }
