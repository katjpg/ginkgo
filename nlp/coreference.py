import re
from typing import Any


PATTERN = re.compile(
    r'^["\']?\s*(our|this|the|these|those)\s+(?:(.*?)\s+)?(\w+)\s*["\']?$',
    re.IGNORECASE,
)


def learn_acronyms(entities: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    Extract acronym definitions from entity texts.

    Detects patterns: "full expansion (ABC)" or "full expansion (or ABC)"
    Learns bidirectional mappings: ABC -> {full, expansion, tokens}
    
    """
    acronym_map = {}

    # pattern matches: "text (ACRONYM)" or "text (or ACRONYM)"
    pattern = re.compile(r"(.+?)\s*\((?:or\s+)?([A-Z]{2,})\)")

    for entity in entities:
        text = entity["extraction_text"]
        match = pattern.search(text)

        if match:
            expansion_text = match.group(1)
            acronym = match.group(2).lower()

            # extract significant tokens from expansion
            tokens = re.findall(r"\b\w+\b", expansion_text.lower())

            # filter short tokens and common words
            stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with"}
            significant = {t for t in tokens if t not in stopwords and len(t) > 1}

            if significant:
                acronym_map[acronym] = significant

    return acronym_map


def normalize_singular(word: str) -> str:
    """
    Convert plural forms to singular using morphological rules.
    """
    if len(word) <= 2:
        return word

    # words ending in 's' that are already singular
    singular_exceptions = {
        "basis",
        "crisis",
        "thesis",
        "analysis",
        "synthesis",
        "hypothesis",
        "emphasis",
        "neurosis",
        "diagnosis",
        "axis",
        "oasis",
        "chassis",
        "canvas",
        "corpus",
        "class",
    }

    if word in singular_exceptions:
        return word

    # irregular plural mappings
    irregular = {
        "data": "datum",
        "criteria": "criterion",
        "phenomena": "phenomenon",
        "analyses": "analysis",
        "theses": "thesis",
        "hypotheses": "hypothesis",
    }

    if word in irregular:
        return irregular[word]

    # regular morphological transformations
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"

    if word.endswith("ves") and len(word) > 3:
        return word[:-3] + "f"

    if word.endswith("es") and len(word) > 3:
        if word.endswith(("sses", "xes", "zes", "ches", "shes")):
            return word[:-2]
        return word[:-2]

    if word.endswith("s") and len(word) > 2:
        return word[:-1]

    return word


def extract_acronyms(text: str) -> set[str]:
    """Extract acronym tokens from parenthetical definitions."""
    acronyms = set()

    pattern = r"\((?:or\s+)?([A-Z]{2,})\)"
    matches = re.findall(pattern, text)
    acronyms.update(m.lower() for m in matches)

    return acronyms


def extract_tokens(text: str, acronym_map: dict[str, set[str]]) -> set[str]:
    """
    Extract and normalize tokens with document-learned acronym expansion.

    Processing pipeline:
        1. Extract acronyms from parentheticals
        2. Remove parenthetical content
        3. Tokenize on word boundaries
        4. Filter stopwords and normalize to singular
        5. Expand acronyms using learned mappings
    """
    # extract acronyms before removing parentheticals
    acronyms = extract_acronyms(text)

    # remove parenthetical content
    cleaned = re.sub(r"\([^)]+\)", "", text)

    # tokenize on word boundaries
    tokens = re.findall(r"\b\w+\b", cleaned.lower())

    # filter stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "our",
        "this",
    }
    significant = {t for t in tokens if t not in stopwords and len(t) > 1}

    # normalize to singular forms
    normalized = {normalize_singular(t) for t in significant}

    # add extracted acronyms
    normalized.update(acronyms)

    # expand acronyms using learned mappings
    expanded = set(normalized)
    for token in normalized:
        if token in acronym_map:
            expanded.update(acronym_map[token])

    return expanded


def compute_overlap(
    generic_text: str, candidate_text: str, acronym_map: dict[str, set[str]]
) -> float:
    """
    Compute normalized lexical overlap between generic reference and candidate.

    Returns value in [0, 1] where higher scores indicate stronger matches.
    """
    generic_tokens = extract_tokens(generic_text, acronym_map)
    candidate_tokens = extract_tokens(candidate_text, acronym_map)

    if not generic_tokens:
        return 0.0

    overlap = len(generic_tokens & candidate_tokens)

    return overlap / len(generic_tokens)


def is_meta_linguistic(text: str) -> bool:
    """
    Identify generic references to the document itself.

    Meta-linguistic references point to the discourse artifact
    (instead of entities described within the discourse)
    """
    meta_terms = {
        "paper",
        "work",
        "study",
        "article",
        "publication",
        "document",
        "manuscript",
        "report",
        "thesis",
        "dissertation",
    }

    # extract tokens without acronym expansion
    tokens = re.findall(r"\b\w+\b", text.lower())
    significant = {t for t in tokens if len(t) > 2}

    return bool(significant & meta_terms)


def search(
    generic_text: str, preceding: list[dict[str, Any]], acronym_map: dict[str, set[str]]
) -> dict[str, Any] | None:
    """
    Find antecedent using lexical overlap with recency bias.

    Algorithm:
        1. Compute overlap score for each candidate in lookback window
        2. Select candidate exceeding threshold with maximum score
        3. Among ties, prefer most recent (recency bias)
    """
    lookback = 10
    window = preceding[-lookback:] if len(preceding) > lookback else preceding

    threshold = 0.3
    best_match = None
    best_score = threshold

    for entity in reversed(window):
        # skip generic entities to avoid transitive chains
        if entity["extraction_class"] == "generic":
            continue

        score = compute_overlap(generic_text, entity["extraction_text"], acronym_map)

        if score > best_score:
            best_score = score
            best_match = entity

    return best_match


def resolve(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Resolve generic entities to antecedents using learned acronym expansions.

    Architecture:
        Phase 1: Learn acronym mappings from entity definitions
        Phase 2: Resolve generic references using token overlap
        Phase 3: Filter meta-linguistic references

    """
    # phase 1: learn acronym expansions from document
    acronym_map = learn_acronyms(entities)

    # phase 2: resolve generic references
    resolved = []

    for i, entity in enumerate(entities):
        if entity["extraction_class"] != "generic":
            resolved.append(entity)
            continue

        # filter meta-linguistic references
        if is_meta_linguistic(entity["extraction_text"]):
            resolved.append(entity)
            continue

        # verify syntactic pattern match
        m = PATTERN.search(entity["extraction_text"])
        if not m:
            resolved.append(entity)
            continue

        # search for antecedent
        antecedent = search(entity["extraction_text"], entities[:i], acronym_map)

        if antecedent:
            # copy-and-replace: antecedent identity + generic position
            resolved_entity = {
                "extraction_class": antecedent["extraction_class"],
                "extraction_text": antecedent["extraction_text"],
                "char_interval": entity["char_interval"],
                "attributes": entity.get("attributes", {}),
            }
            resolved.append(resolved_entity)
        else:
            resolved.append(entity)

    return resolved
