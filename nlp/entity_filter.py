from collections import Counter
from pydantic import BaseModel, Field
from spacy.tokens import Doc
from typing import Literal
import spacy
import networkx as nx
import re


nlp = spacy.load("en_core_web_sm")
STOPWORDS = nlp.Defaults.stop_words

INFRA = {
    "value", "key", "query", "output", "input",
    "sequence", "layer", "vector", "token",
    "parameter", "weight", "hidden state",
    "representation", "embedding",
    "example", "result", "performance",
    "model", "method", "approach",
    "system", "architecture",
    "data", "dataset", "feature",
    "information", "detail", "aspect", "element", "component",
    "symbol", "position",
    "encoder", "decoder", "attention", "mechanism",
    "second", "minute", "hour", "step", "iteration", "epoch", "day",
}

GENERIC_PATTERNS = [
    re.compile(r"^the [a-z]+$"),
    re.compile(r"^our [a-z]+s?$"),
    re.compile(r"^this [a-z]+$"),
    re.compile(r"^these [a-z]+s$"),
    re.compile(r"^each [a-z]+$"),
    re.compile(r"^all [a-z]+s?$"),
    re.compile(r"^every [a-z]+$"),
    re.compile(r"^[a-z]+ of [a-z]+$"),
    re.compile(r"^\w{1}$"),
    re.compile(r"^[α-ωΑ-Ω]\s*\d*$"),
]

CONTRASTIVE_TERMS = {
    "input", "output", "source", "target", "left", "right", "forward", "backward",
    "encoder", "decoder", "top", "bottom", "first", "last", "begin", "end",
    "english", "german", "french", "chinese", "spanish", "arabic", "japanese",
    "train", "test", "validation", "dev",
}


EntityType = Literal["method", "task", "dataset", "metric", "object", "other"]


class FilterConfig(BaseModel):
    """Configuration for entity filtering pipeline."""
    min_freq: int = Field(default=1, ge=1)
    exclude_other: bool = True
    use_fuzzy: bool = True
    top_k: int | None = None
    jaccard_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    window_size: int = Field(default=3, ge=1)
    pagerank_alpha: float = Field(default=0.85, ge=0.0, le=1.0)


def lemmatize_word(word: str) -> str:
    """Get lemma form of word using spaCy."""
    doc = nlp(word)
    return doc[0].lemma_ if doc else word


def is_infra_term(text: str) -> bool:
    """Check if text consists only of infrastructure terms."""
    words = text.lower().split()
    for word in words:
        lemma = lemmatize_word(word)
        if lemma not in INFRA and word not in INFRA:
            return False
    return True


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing hyphens, and lemmatizing plurals."""
    normalized = " ".join(text.lower().strip().replace("-", " ").split())
    
    words = []
    for word in normalized.split():
        if (word.endswith("s") and len(word) > 3 and 
            not word.endswith(("ss", "us", "ous", "ness", "ics", "sis", "ysis", "itis", "ess", "less"))):
            words.append(word[:-1])
        else:
            words.append(word)
    
    return " ".join(words)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard coefficient between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def has_contrastive_difference(words1: set, words2: set) -> bool:
    """Check if word sets differ in contrastive terms."""
    diff = (words1 - words2) | (words2 - words1)
    return bool(diff & CONTRASTIVE_TERMS)


def should_merge(entity1: dict, entity2: dict) -> bool:
    """Determine if entities should merge using Jaccard similarity."""
    if entity1["type"] != entity2["type"]:
        return False
    
    norm1 = normalize_text(entity1["text"])
    norm2 = normalize_text(entity2["text"])
    
    if norm1 == norm2:
        return True
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    similarity = jaccard_similarity(words1, words2)
    
    if similarity < 0.6:
        return False
    
    if has_contrastive_difference(words1, words2):
        return False
    
    if words1.issubset(words2) or words2.issubset(words1):
        return True
    
    return similarity >= 0.75


def select_representative(entities: list[dict]) -> dict:
    """Select best entity from duplicate cluster."""
    if len(entities) == 1:
        return entities[0]
    
    def sort_key(e):
        return (-e.get("pr_score", 0), -len(e["text"]))
    
    return sorted(entities, key=sort_key)[0]


def dedupe_fuzzy(entities: list[dict]) -> list[dict]:
    """Remove fuzzy duplicates using Jaccard similarity and union-find."""
    if not entities:
        return []
    
    by_type = {}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in by_type:
            by_type[entity_type] = []
        by_type[entity_type].append(entity)
    
    unique_entities = []
    
    for entity_type, type_entities in by_type.items():
        n = len(type_entities)
        parent = list(range(n))
        
        def find(x):
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Union by root."""
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge(type_entities[i], type_entities[j]):
                    union(i, j)
        
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(type_entities[i])
        
        for cluster_entities in clusters.values():
            best = select_representative(cluster_entities)
            unique_entities.append(best)
    
    return sorted(unique_entities, key=lambda e: e.get("pr_score", 0), reverse=True)


def dedupe_semantic(entities: list[dict], use_fuzzy: bool = True) -> list[dict]:
    """Remove duplicates using normalization and fuzzy matching."""
    seen = {}
    unique = []
    
    for entity in entities:
        text = entity["text"]
        normalized = normalize_text(text)
        score = entity.get("pr_score", 0)
        
        if normalized in seen:
            existing = seen[normalized]
            existing_score = existing.get("pr_score", 0)
            
            if score > existing_score:
                unique.remove(existing)
                seen[normalized] = entity
                unique.append(entity)
        else:
            seen[normalized] = entity
            unique.append(entity)
    
    if use_fuzzy and len(unique) > 1:
        unique = dedupe_fuzzy(unique)
    
    return unique


def is_notation_variable(text: str) -> bool:
    """Check if text is mathematical notation."""
    text_normalized = ' '.join(text.split())
    
    if len(text_normalized) == 1:
        return True
    
    tokens = text_normalized.split()
    
    if all(len(t) <= 2 or (len(t) <= 5 and t[0].isupper()) for t in tokens):
        if len(tokens[0]) <= 2:
            return True
    
    if len(tokens) == 2 and len(tokens[0]) <= 2 and len(tokens[1]) <= 2:
        return True
    
    return False


def is_parsing_artifact(text: str) -> bool:
    """Check if text is likely a parsing error."""
    if len(text.split()) == 2:
        first, second = text.split()
        if len(first) == 1 and second.lower() in ["model", "drop", "score", "layer", "value", "rate"]:
            return True
    
    if "  " in text:
        return True
    
    if text.isupper() and len(text) <= 2 and len(text.split()) == 1:
        return True
    
    if len(text) == 1:
        return True
    
    return False


def is_incomplete_phrase(text: str) -> bool:
    """Check if text is incomplete using POS tags and dependency patterns."""
    words = text.split()
    
    if len(words) != 2:
        return False
    
    doc_snippet = nlp(text)
    tags = [token.pos_ for token in doc_snippet]
    
    if tags == ["ADJ", "ADJ"]:
        return True
    
    # check if last token is ADJ modifying a missing noun
    if len(doc_snippet) == 2:
        last_token = doc_snippet[-1]
        if last_token.pos_ == "ADJ":
            return True
    
    # check if phrase ends with prefix/modifier expecting continuation
    # e.g., "multi-head" without "attention"
    last_word = words[-1].lower()
    if last_word in {"multi", "self", "cross", "pre", "post", "sub"}:
        return True
    
    return False



def is_valid_metric(text: str) -> bool:
    """Check if metric has substantive content."""
    text_lower = text.lower()
    
    if is_notation_variable(text) or is_parsing_artifact(text):
        return False
    
    standard_metrics = {
        "bleu", "rouge", "meteor", "perplexity", "f1", "precision", "recall",
        "map", "ndcg", "auc", "loss", "error rate", "bleu score", "rouge-l",
        "mse", "rmse", "mae"
    }
    
    if text_lower in standard_metrics:
        return True
    
    if any(c.isdigit() for c in text):
        return True
    
    vague_single_words = {"accuracy", "score", "error", "rate", "value", "measure"}
    if len(text.split()) == 1 and text_lower in vague_single_words:
        return False
    
    return True


def filter_lexical(entities: list[dict]) -> list[dict]:
    """Remove stopwords and generic patterns."""
    filtered = []
    
    for entity in entities:
        text = entity["text"]
        text_lower = text.lower()
        words = text.split()
        
        if len(text) < 3:
            continue
        if text_lower in STOPWORDS:
            continue
        if text_lower.startswith(("a ", "an ", "the ")):
            continue
        if len(words) == 1 and is_infra_term(text_lower):
            continue
        if any(pattern.match(text_lower) for pattern in GENERIC_PATTERNS):
            continue
        if len(words) > 7:
            continue
        if text_lower in {"things", "cases", "ways", "types", "forms"}:
            continue
        if is_parsing_artifact(text):
            continue
        if is_notation_variable(text):
            continue
        if is_incomplete_phrase(text):
            continue
        
        if entity["type"] == "metric":
            if not is_valid_metric(text):
                continue
        
        if entity["type"] == "object":
            if is_infra_term(text_lower):
                continue
        
        filtered.append(entity)
    
    return filtered


def filter_type(entities: list[dict], exclude_other: bool = True) -> list[dict]:
    """Keep meaningful entity types."""
    valid = {"method", "task", "dataset", "metric", "model"}
    
    if not exclude_other:
        valid.add("other")
    
    domain_objs = []
    for e in entities:
        if e["type"] == "object":
            text_lower = e["text"].lower()
            if not is_infra_term(text_lower) and len(e["text"].split()) >= 2:
                domain_objs.append(e)
    
    return [e for e in entities if e["type"] in valid] + domain_objs


def filter_freq(entities: list[dict], min_count: int = 1, strict_objects: bool = True) -> list[dict]:
    """Keep entities mentioned at least min_count times."""
    counts = Counter(e["text"] for e in entities)
    filtered = []
    
    for entity in entities:
        count = counts[entity["text"]]
        
        if strict_objects and entity["type"] == "object":
            if count >= 2:
                filtered.append(entity)
        else:
            if count >= min_count:
                filtered.append(entity)
    
    return filtered


def build_graph(entities: list[dict], doc: Doc, window_size: int = 3) -> nx.Graph:
    """Build entity co-occurrence graph."""
    G = nx.Graph()
    entity_set = {e["text"].lower() for e in entities}
    
    entity_positions = {}
    for sent_idx, sent in enumerate(doc.sents):
        sent_text = sent.text.lower()
        for entity_text in entity_set:
            if entity_text in sent_text:
                if entity_text not in entity_positions:
                    entity_positions[entity_text] = []
                entity_positions[entity_text].append(sent_idx)
    
    sents = list(doc.sents)
    for i in range(0, len(sents), window_size):
        window_range = range(i, min(i + window_size, len(sents)))
        
        window_entities = [
            e_text for e_text, positions in entity_positions.items()
            if any(pos in window_range for pos in positions)
        ]
        
        for j, e1 in enumerate(window_entities):
            for e2 in window_entities[j+1:]:
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)
    
    return G


def rank_pagerank(entities: list[dict], doc: Doc) -> list[dict]:
    """Rank entities by PageRank."""
    if len(entities) <= 1:
        for e in entities:
            e["pr_score"] = 1.0
        return entities
    
    G = build_graph(entities, doc, window_size=3)
    
    if len(G.nodes()) == 0:
        for e in entities:
            e["pr_score"] = 0.0
        return entities
    
    scores = nx.pagerank(G, weight="weight", alpha=0.85)
    
    type_weights = {"method": 1.2, "task": 1.15, "dataset": 1.0, "metric": 1.05, "object": 0.8}
    
    counts = Counter(e["text"] for e in entities)
    max_count = max(counts.values()) if counts else 1
    
    for entity in entities:
        base_score = scores.get(entity["text"].lower(), 0.0)
        type_weight = type_weights.get(entity["type"], 1.0)
        
        if base_score == 0.0:
            freq_score = counts[entity["text"]] / max_count * 0.01
            entity["pr_score"] = freq_score * type_weight
        else:
            entity["pr_score"] = base_score * type_weight
    
    return sorted(entities, key=lambda e: e["pr_score"], reverse=True)


def filter_pagerank(entities: list[dict], doc: Doc, top_k: int | None = None) -> list[dict]:
    """Filter entities by PageRank."""
    ranked = rank_pagerank(entities, doc)
    
    if top_k is not None:
        return ranked[:top_k]
    
    return ranked


def filter_pipeline(
    entities: list[dict], 
    doc: Doc,
    min_freq: int = 1,
    exclude_other: bool = True,
    use_fuzzy: bool = True,
    top_k: int | None = None
) -> list[dict]:
    """Entity filtering pipeline."""
    entities = filter_lexical(entities)
    entities = filter_type(entities, exclude_other=exclude_other)
    entities = filter_freq(entities, min_count=min_freq, strict_objects=True)
    entities = filter_pagerank(entities, doc, top_k=None)
    entities = dedupe_semantic(entities, use_fuzzy=use_fuzzy)
    
    if top_k is not None:
        entities = entities[:top_k]
    
    return entities


def analyze_impact(original: list[dict], filtered: list[dict]) -> dict:
    """Compare entity distributions before and after filtering."""
    orig_counts = Counter(e["type"] for e in original)
    filt_counts = Counter(e["type"] for e in filtered)
    
    return {
        "original_count": len(original),
        "filtered_count": len(filtered),
        "reduction_pct": (1 - len(filtered) / len(original)) * 100 if original else 0,
        "type_distribution_before": dict(orig_counts),
        "type_distribution_after": dict(filt_counts),
        "unique_before": len(set(e["text"] for e in original)),
        "unique_after": len(set(e["text"] for e in filtered)),
    }
