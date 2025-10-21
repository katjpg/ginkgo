import re

def clean_citations(text: str) -> str:
    """Remove inline year citations and footnote markers."""
    text = re.sub(r'\(\d{4}[a-z]?(?:;\s*\d{4}[a-z]?)*\)', '', text)
    text = re.sub(r'foot_\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

