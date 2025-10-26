from typing import Any
from itertools import combinations
from dataclasses import dataclass

from spacy.tokens import Doc, Span

from nlp.syntactic import find_sdp, get_1hop, verbalize_path, convert_entity


@dataclass
class EntityPair:
    """Entity pair with aggregated mentions across document."""

    head: dict[str, Any]
    tail: dict[str, Any]
    mentions: list[dict[str, Any]]


VALID_TYPE_PAIRS = {
    # method relations
    ("METHOD", "TASK"),  # "BERT for NER"
    ("METHOD", "DATASET"),  # "ResNet on ImageNet"
    ("METHOD", "METRIC"),  # "BERT achieves F1 score"
    ("METHOD", "OBJECT"),  # "attention mechanism for proteins"
    ("METHOD", "METHOD"),  # "BERT outperforms LSTM"
    ("METHOD", "OTHER"),  # "BERT uses attention mechanism"
    # task relations
    ("TASK", "DATASET"),  # "classification on ImageNet"
    ("TASK", "METRIC"),  # "NER evaluated by F1"
    ("TASK", "OBJECT"),  # "classification of proteins"
    # dataset relations
    ("DATASET", "METRIC"),  # "ImageNet accuracy"
    ("DATASET", "OBJECT"),  # "CoNLL2003 for entities"
    # metric relations
    ("METRIC", "OBJECT"),  # "accuracy of predictions"
    # generic (anaphoric) can pair with anything
    ("GENERIC", "METHOD"),
    ("GENERIC", "TASK"),
    ("GENERIC", "DATASET"),
    ("GENERIC", "OBJECT"),
    ("GENERIC", "METRIC"),
    ("GENERIC", "OTHER"),
    # other (technical concepts)
    ("OTHER", "METHOD"),  # "architecture uses transformer"
    ("OTHER", "TASK"),  # "mechanism for classification"
    ("OTHER", "OBJECT"),  # "layer for features"
}


def create_pairs(entities: list[dict], doc: Doc) -> list[dict]:
    """Generate entity pairs with syntactic features.

    Args:
        entities: List of entity dictionaries with text, type, and char_interval
        doc: Parsed spaCy document

    Returns:
        List of entity pair dictionaries with syntactic features
    """
    pairs = []

    # generate all possible entity combinations
    for e1, e2 in combinations(entities, 2):
        span1 = convert_entity(e1, doc)
        span2 = convert_entity(e2, doc)

        # skip if entities couldn't be converted to spans
        if not span1 or not span2:
            continue

        # only process entities within same sentence for syntactic features
        if span1.sent != span2.sent:
            continue

        # extract syntactic features
        t1 = span1.root
        t2 = span2.root

        sdp = find_sdp(t1, t2)
        context = get_1hop(sdp)

        pairs.append(
            {
                "head": e1,
                "tail": e2,
                "sdp_tokens": [t.text for t in sdp],
                "context_tokens": [t.text for t in context],
                "syntax": verbalize_path(t1, t2),
                "sentence": span1.sent.text,
            }
        )

    return pairs


def filter_by_type(pairs: list[dict]) -> list[dict]:
    """Filter pairs by valid entity type combinations.

    Args:
        pairs: List of entity pair dictionaries

    Returns:
        Filtered list containing only semantically plausible type combinations
    """
    filtered = []

    for pair in pairs:
        type_tuple = (pair["head"]["type"], pair["tail"]["type"])

        # only check forward direction - relations are directional
        if type_tuple in VALID_TYPE_PAIRS:
            filtered.append(pair)

    return filtered


def aggregate_mentions(pairs: list[dict]) -> list[EntityPair]:
    """Group multiple mentions of same entity pair.

    Args:
        pairs: List of entity pair dictionaries

    Returns:
        List of EntityPair objects with aggregated mentions
    """
    grouped = {}

    for pair in pairs:
        # create canonical key from entity texts
        key = (pair["head"]["text"], pair["tail"]["text"])

        if key not in grouped:
            grouped[key] = EntityPair(head=pair["head"], tail=pair["tail"], mentions=[])

        # aggregate syntactic evidence from this mention
        grouped[key].mentions.append(
            {
                "sentence": pair["sentence"],
                "syntax": pair["syntax"],
                "context": pair.get("context_tokens", []),
                "sdp": pair.get("sdp_tokens", []),
            }
        )

    return list(grouped.values())


def filter_by_distance(pairs: list[dict], doc: Doc, max_dist: int = 3) -> list[dict]:
    """Filter cross-sentence pairs by sentence distance.

    Args:
        pairs: List of entity pair dictionaries
        doc: Parsed spaCy document
        max_dist: Maximum sentence distance between entities

    Returns:
        Filtered list with entities within max_dist sentences
    """
    filtered = []

    # build sentence index for efficient lookup
    sentences = list(doc.sents)
    sent_lookup = {sent: idx for idx, sent in enumerate(sentences)}

    for pair in pairs:
        span1 = convert_entity(pair["head"], doc)
        span2 = convert_entity(pair["tail"], doc)

        if not span1 or not span2:
            continue

        # calculate sentence distance
        s1_idx = sent_lookup.get(span1.sent)
        s2_idx = sent_lookup.get(span2.sent)

        if s1_idx is not None and s2_idx is not None:
            distance = abs(s1_idx - s2_idx)

            if distance <= max_dist:
                filtered.append(pair)

    return filtered


def find_repeat_pairs(entities: list[dict], doc: Doc, min_count: int = 2) -> list[dict]:
    """Find entity pairs appearing multiple times across document.

    Args:
        entities: List of entity dictionaries
        doc: Parsed spaCy document
        min_count: Minimum co-occurrence count

    Returns:
        List of frequently co-occurring entity pairs
    """
    entity_sents = {}

    # map entities to their containing sentences
    for entity in entities:
        span = convert_entity(entity, doc)
        if span:
            key = entity["text"]
            if key not in entity_sents:
                entity_sents[key] = []
            entity_sents[key].append(span.sent.text)

    pairs = []

    # find entities that frequently co-occur in same sentences
    for e1_text, e1_sents in entity_sents.items():
        for e2_text, e2_sents in entity_sents.items():
            # avoid duplicate pairs and self-pairs
            if e1_text >= e2_text:
                continue

            # count co-occurrences
            count = sum(1 for s in e1_sents if s in e2_sents)

            if count >= min_count:
                pairs.append(
                    {
                        "head_text": e1_text,
                        "tail_text": e2_text,
                        "count": count,
                        "sentences": list(set(e1_sents) & set(e2_sents)),
                    }
                )

    return pairs
