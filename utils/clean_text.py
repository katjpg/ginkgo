import re


def clean_citations(text: str) -> str:
    """Remove inline year citations and footnote markers."""
    text = re.sub(r"\(\d{4}[a-z]?(?:;\s*\d{4}[a-z]?)*\)", "", text)
    text = re.sub(r"foot_\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_concatenation(text: str) -> str:
    """Fix words concatenated by camelCase (e.g., 'ofApplications' -> 'of Applications')."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def remove_figure_refs(text: str) -> str:
    """Remove figure and table references."""
    text = re.sub(r"Figure\s*\d+", "", text)
    text = re.sub(r"Table\s*\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text: str) -> str:
    """Remove URLs and web addresses."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_algorithms(text: str) -> str:
    """Remove algorithm pseudocode patterns."""
    text = re.sub(r"Algorithm\s+\d+:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()
