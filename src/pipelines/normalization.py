import re
import unicodedata

__all__ = ["normalize_text"]

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    text = re.sub(r'\n?\s*page\s+\d+\s*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n-{2,}\n', '\n', text)

    text = re.sub(r'-\n', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()
