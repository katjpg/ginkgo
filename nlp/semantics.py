from dataclasses import dataclass
from typing import Any
import re
from spacy.tokens import Doc, Span, Token


@dataclass
class SemanticRelation:
    """A semantic relation found by a linguistic pattern."""

    relation_type: str
    source_span: Span
    target_span: Span
    confidence: float
    provenance: str


def validate_entity_span(span: Span, entity_type: str) -> bool:
    """
    Filter out spans that are not meaningful entities.

    Ensures the span is not a function word (e.g., 'the', 'is') or a
    placeholder (e.g., 'generic' type).
    """
    # reject single-token function words
    if len(span) == 1:
        token = span[0]
        if token.pos_ in ("PRON", "DET", "AUX", "CCONJ", "SCONJ"):
            return False

    # validate span head is content word
    head = span.root
    if head.pos_ in ("PRON", "DET", "AUX", "CCONJ", "SCONJ", "ADP"):
        return False

    # reject generic entity classifications
    generic_types = {"generic"}
    if entity_type.lower() in generic_types:
        return False

    return True


def has_punctuation_marker(token1: Token, token2: Token, doc: Doc) -> bool:
    """
    Check for appositive punctuation (comma, dash, parentheses) between tokens.
    """
    start_idx = min(token1.i, token2.i)
    end_idx = max(token1.i, token2.i)

    # scan tokens between the two heads
    for i in range(start_idx + 1, end_idx):
        token = doc[i]
        if token.pos_ == "PUNCT":
            if token.text in (",", "—", "-", "(", ")"):
                return True

    return False


def find_appositive_relations(
    spans: list[tuple[dict[str, Any], Span]], doc: Doc
) -> list[SemanticRelation]:
    """
    Find EQUIVALENT_TO relations from appositive constructions.

    Detects syntactic appositives (e.g., "Our system, X,...") using
    the 'appos' dependency.
    """
    relations = []

    # build token-to-entity mapping
    token_to_entity = {}
    for entity, span in spans:
        for token in span:
            token_to_entity[token] = (entity, span)

    for entity, span in spans:
        if not validate_entity_span(span, entity["extraction_class"]):
            continue

        root = span.root

        # check appositive children
        for child in root.children:
            if child.dep_ == "appos":
                if child in token_to_entity:
                    appos_entity, appos_span = token_to_entity[child]

                    if not validate_entity_span(
                        appos_span, appos_entity["extraction_class"]
                    ):
                        continue

                    if appos_span != span:
                        # confidence based on punctuation markers
                        if has_punctuation_marker(root, child, doc):
                            confidence = 0.95
                        else:
                            distance = abs(child.i - root.i)
                            confidence = 0.90 if distance <= 3 else 0.80

                        relations.append(
                            SemanticRelation(
                                relation_type="EQUIVALENT_TO",
                                source_span=span,
                                target_span=appos_span,
                                confidence=confidence,
                                provenance="semantic_appositive",
                            )
                        )

        # check if root is appositive
        if root.dep_ == "appos" and root.head in token_to_entity:
            head_entity, head_span = token_to_entity[root.head]

            if not validate_entity_span(head_span, head_entity["extraction_class"]):
                continue

            if head_span != span:
                if has_punctuation_marker(root.head, root, doc):
                    confidence = 0.95
                else:
                    distance = abs(root.i - root.head.i)
                    confidence = 0.90 if distance <= 3 else 0.80

                relations.append(
                    SemanticRelation(
                        relation_type="EQUIVALENT_TO",
                        source_span=head_span,
                        target_span=span,
                        confidence=confidence,
                        provenance="semantic_appositive",
                    )
                )

    return relations


def has_categorical_prep_phrase(
    attribute_token: Token, categorical_markers: set[str]
) -> bool:
    """
    Check for taxonomic markers like "type of" or "kind of".

    Used by the copular finder to distinguish true IS_A relations
    (e.g., "is a type of network") from simple attributes (e.g., "is fast").

    Example: "type of network" -> True
             "system that processes types" -> False
    """
    for child in attribute_token.children:
        # check for prepositional phrase headed by "of"
        if child.dep_ == "prep" and child.lemma_ == "of":
            # check if prep has object containing categorical marker
            for prep_child in child.children:
                if prep_child.dep_ == "pobj":
                    if prep_child.lemma_ in categorical_markers:
                        return True

        # also check direct children for categorical markers
        if child.lemma_ in categorical_markers:
            # ensure it's in appropriate dependency relation
            if child.dep_ in ("compound", "amod"):
                return True

    # check if attribute token itself is categorical marker
    if attribute_token.lemma_ in categorical_markers:
        return True

    return False


def find_copular_relations(
    spans: list[tuple[dict[str, Any], Span]], doc: Doc
) -> list[SemanticRelation]:
    """
    Find IS_A relations from copular verbs (e.g., 'is', 'be').

    Looks for patterns like "X is a type of Y" using the
    `has_categorical_prep_phrase` helper to ensure it's a true
    taxonomic link.
    """
    relations = []

    # build token-to-entity mapping
    token_to_entity = {}
    for entity, span in spans:
        for token in span:
            token_to_entity[token] = (entity, span)

    copular_verbs = {"be", "remain", "become", "constitute"}
    categorical_markers = {
        "type",
        "kind",
        "form",
        "class",
        "variety",
        "instance",
        "example",
        "category",
    }

    for token in doc:
        if token.lemma_ not in copular_verbs or token.pos_ != "VERB":
            continue

        # filter negation
        if any(child.dep_ == "neg" for child in token.children):
            continue

        # extract subject and attribute
        subject_token = None
        attribute_token = None

        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject_token = child
            elif child.dep_ == "attr":
                attribute_token = child

        if not subject_token or not attribute_token:
            continue

        # validate both are nouns
        if subject_token.pos_ not in ("NOUN", "PROPN"):
            continue
        if attribute_token.pos_ not in ("NOUN", "PROPN"):
            continue

        # apply scoped categorical marker detection
        if not has_categorical_prep_phrase(attribute_token, categorical_markers):
            continue

        # map to entity spans
        subject_info = None
        attribute_info = None

        if subject_token in token_to_entity:
            subject_info = token_to_entity[subject_token]
        if attribute_token in token_to_entity:
            attribute_info = token_to_entity[attribute_token]

        if subject_info and attribute_info:
            subject_entity, subject_span = subject_info
            attribute_entity, attribute_span = attribute_info

            if not validate_entity_span(
                subject_span, subject_entity["extraction_class"]
            ):
                continue
            if not validate_entity_span(
                attribute_span, attribute_entity["extraction_class"]
            ):
                continue

            if subject_span != attribute_span:
                relations.append(
                    SemanticRelation(
                        relation_type="IS_A",
                        source_span=subject_span,
                        target_span=attribute_span,
                        confidence=0.90,
                        provenance="semantic_copular",
                    )
                )

    return relations


def validate_acronym(acronym: str, expansion: str) -> bool:
    """
    Check if an acronym (e.g., 'CNN') matches an expansion.

    Implements the Schwartz-Hearst algorithm for:
    - Leading articles (e.g., 'a', 'the')
    - Internal function words
    - Hyphenated terms
    """
    # filter leading articles
    expansion_words = expansion.split()
    filtered_words = [w for w in expansion_words if w.lower() not in ("a", "an", "the")]

    if not filtered_words:
        return False

    # extract initial letters from content words
    first_letters = "".join(
        w[0].upper() for w in filtered_words if w and w[0].isalpha()
    )

    # normalize acronym for comparison
    acronym_clean = "".join(c.upper() for c in acronym if c.isalpha())

    if not acronym_clean:
        return False

    # exact match (highest confidence)
    if acronym_clean == first_letters:
        return True

    # subsequence match (allows intermediate words)
    if len(acronym_clean) <= len(first_letters):
        acronym_idx = 0
        for letter in first_letters:
            if (
                acronym_idx < len(acronym_clean)
                and letter == acronym_clean[acronym_idx]
            ):
                acronym_idx += 1

        # require first character match (Schwartz-Hearst constraint)
        if acronym_idx == len(acronym_clean) and acronym_clean[0] == first_letters[0]:
            return True

    return False


def find_acronym_relations(
    entities: list[dict[str, Any]], spans: list[tuple[dict[str, Any], Span]]
) -> list[SemanticRelation]:
    """
    Find acronyms defined in text, like "Expansion (Acronym)".

    Correctly assigns relation direction:
    - 'CNN' -> 'Convolutional...' (ABBREVIATES)
    - 'Expansion (Acronym)' -> 'Expansion' (EQUIVALENT_TO)
    """
    relations = []

    acronym_pattern = re.compile(r"(.+?)\s*\((?:or\s+)?([A-Z0-9][A-Z0-9\-]*)\)")
    span_by_bounds = {(s.start, s.end): (e, s) for e, s in spans}

    for entity, span in spans:
        text = entity["extraction_text"]
        match = acronym_pattern.search(text)

        if not match:
            continue

        expansion_text = match.group(1).strip()
        acronym = match.group(2)

        if not validate_acronym(acronym, expansion_text):
            continue

        # confidence calculation
        expansion_words = [
            w for w in expansion_text.split() if w.lower() not in ("a", "an", "the")
        ]
        first_letters = "".join(
            w[0].upper() for w in expansion_words if w and w[0].isalpha()
        )
        acronym_clean = "".join(c.upper() for c in acronym if c.isalpha())

        confidence = 0.95 if acronym_clean == first_letters else 0.85

        # search for related entities
        for other_entity, other_span in spans:
            if other_span == span:
                continue

            other_text = other_entity["extraction_text"]

            # Case 1: isolated acronym (correct abbreviation relation)
            if other_text == acronym:
                relations.append(
                    SemanticRelation(
                        relation_type="ABBREVIATES",
                        source_span=other_span,  # short form
                        target_span=span,  # long form
                        confidence=confidence,
                        provenance="semantic_acronym",
                    )
                )

            # Case 2: isolated expansion (textual equivalence, not abbreviation)
            elif expansion_text == other_text or (
                expansion_text in other_text and len(other_text) < len(text)
            ):
                relations.append(
                    SemanticRelation(
                        relation_type="EQUIVALENT_TO",  # Changed from ABBREVIATES
                        source_span=span,  # full form with annotation
                        target_span=other_span,  # expansion only
                        confidence=confidence,
                        provenance="semantic_acronym",
                    )
                )

    return relations


def extract_semantic_relations(
    entities: list[dict[str, Any]], spans: list[tuple[dict[str, Any], Span]], doc: Doc
) -> list[dict[str, Any]]:
    """

    Main function to find all pattern-based semantic relations.

    Runs the appositive, copular, and acronym extractors and
    formats their results for output.
    """
    semantic_relations = []

    # detect patterns
    appositive_relations = find_appositive_relations(spans, doc)
    semantic_relations.extend(appositive_relations)

    copular_relations = find_copular_relations(spans, doc)
    semantic_relations.extend(copular_relations)

    acronym_relations = find_acronym_relations(entities, spans)
    semantic_relations.extend(acronym_relations)

    # convert to dictionary format
    output_relations = []
    span_to_entity = {(s.start, s.end): e for e, s in spans}

    for rel in semantic_relations:
        source_key = (rel.source_span.start, rel.source_span.end)
        target_key = (rel.target_span.start, rel.target_span.end)

        source_entity = span_to_entity.get(source_key)
        target_entity = span_to_entity.get(target_key)

        if source_entity and target_entity:
            output_relations.append(
                {
                    "source": source_entity["extraction_text"],
                    "source_type": source_entity["extraction_class"],
                    "relation": rel.relation_type,
                    "target": target_entity["extraction_text"],
                    "target_type": target_entity["extraction_class"],
                    "confidence": rel.confidence,
                    "provenance": rel.provenance,
                }
            )

    return output_relations
