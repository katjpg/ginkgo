import re
from collections import Counter


def dehyphenate(text: str) -> str:
    """Remove line-break hyphens from PDF parsing."""
    text = re.sub(r'(\w+)-\s*\n\s*([a-z])', r'\1\2', text)
    text = re.sub(r'(\w+)- ([a-z])', r'\1\2', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Fix irregular spacing from PDF parsing."""
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    return text.strip()


def fix_concatenation(text: str) -> str:
    """Fix parsing concatenation errors while preserving intentional camelCase."""
    # fix doubled capitals at word start
    text = re.sub(r'\b([A-Z])\1([a-z])', r'\1\2', text)
    
    # fix concatenated words using capturing group instead of variable-width lookbehind
    # captures lowercase sequence and capital letter, adds space between them
    text = re.sub(r'([a-z]{4,})([A-Z][a-z])', r'\1 \2', text)
    
    return text


def fix_repeated_capitals(text: str, known_terms: set[str] | None = None) -> str:
    """Fix doubled capital letters from parsing errors."""
    if known_terms is None:
        known_terms = set()
    
    def replace_doubled(match):
        full_word = match.group(0)
        if full_word in known_terms:
            return full_word
        return re.sub(r'([A-Z])\1', r'\1', full_word)
    
    return re.sub(r'\b[A-Z][A-Za-z]+\b', replace_doubled, text)


def clean_punctuation_artifacts(text: str) -> str:
    """Remove punctuation artifacts from text parsing."""
    text = re.sub(r'[;,]\s*[;,]+', ',', text)
    text = re.sub(r'\s+[;,]\s+', ' ', text)
    text = re.sub(r'\(\s*[;,]+\s*\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_urls(text: str) -> str:
    """Remove URLs and web addresses."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_repeated_text(text: str, threshold_ratio: float = 0.1) -> str:
    """Remove repeated headers/footers from multi-page extraction."""
    lines = text.split('\n')
    line_counts = Counter(lines)
    threshold = max(1, int(len(lines) * threshold_ratio))
    cleaned = [line for line in lines if line_counts[line] <= threshold or line.strip() == '']
    return '\n'.join(cleaned)


def resolve_abbreviations(text: str, abbrev_dict: dict[str, str]) -> str:
    """Resolve common abbreviations using a lookup dict."""
    for abbrev, full_form in abbrev_dict.items():
        text = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, text)
    return text


def preprocess_section(
    text: str, 
    abbrev_dict: dict[str, str] | None = None, 
    known_terms: set[str] | None = None
) -> str:
    """Apply minimal preprocessing pipeline for scientific text extraction."""
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    text = fix_concatenation(text)
    text = fix_repeated_capitals(text, known_terms)
    text = clean_punctuation_artifacts(text)
    text = remove_urls(text)
    
    if abbrev_dict is not None:
        text = resolve_abbreviations(text, abbrev_dict)
    
    return text


def preprocess_document(
    text: str, 
    remove_repeated: bool = True, 
    abbrev_dict: dict[str, str] | None = None, 
    known_terms: set[str] | None = None
) -> str:
    """Full document preprocessing with optional header/footer removal."""
    if remove_repeated:
        text = remove_repeated_text(text)
    
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    text = fix_concatenation(text)
    text = fix_repeated_capitals(text, known_terms)
    text = clean_punctuation_artifacts(text)
    text = remove_urls(text)
    
    if abbrev_dict is not None:
        text = resolve_abbreviations(text, abbrev_dict)
    
    return text