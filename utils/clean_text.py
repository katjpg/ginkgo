import re


def clean_citations(text: str) -> str:
    """Remove inline year citations and footnote markers."""
    text = re.sub(r"\(\d{4}[a-z]?(?:;\s*\d{4}[a-z]?)*\)", "", text)
    text = re.sub(r"foot_\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_concatenation(text: str) -> str:
    """Fix camelCase word concatenation (e.g., 'ofApplications' -> 'of Applications')."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def remove_figure_refs(text: str) -> str:
    """Remove figure and table references."""
    text = re.sub(r"Figure\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Table\s*\d+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text: str) -> str:
    """Remove URLs and web addresses."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def clean_punctuation_artifacts(text: str) -> str:
    """Remove punctuation artifacts from text cleaning."""
    text = re.sub(r"[;,]\s*[;,]+", ",", text)
    text = re.sub(r"\s+[;,]\s+", " ", text)
    text = re.sub(r"\(\s*[;,]+\s*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text: str) -> str:
    """Remove bullet points and special characters."""
    text = re.sub(r"[•◦▪▫⁃‣⦾⦿]", "", text)
    text = re.sub(r"[\u2022\u2023\u2043\u204C\u204D]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_algorithms(text: str) -> str:
    """Remove algorithm pseudocode blocks (use sparingly - may remove lemmas)."""
    text = re.sub(r"Algorithm\s+\d+:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_section(text: str) -> str:
    """Apply standard preprocessing pipeline for section text."""
    text = fix_concatenation(text)
    text = clean_citations(text)
    text = clean_punctuation_artifacts(text)
    text = remove_figure_refs(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    return text
