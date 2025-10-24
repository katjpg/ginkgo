from dataclasses import dataclass
from typing import Any, Optional
import spacy
from spacy.tokens import Doc, Span, Token

nlp = spacy.load("en_core_web_sm")


@dataclass
class VerbRelation:
    """A verb-mediated relation (Subject-Verb-Object)."""

    verb: str
    subject_span: Span
    object_span: Span
    negated: bool
    modal: Optional[str]

    def matches(self, s1: Span, s2: Span) -> tuple[Span, Span] | None:
        """Return (source, target) if spans match semantic roles."""
        if self.subject_span == s1 and self.object_span == s2:
            return (s1, s2)
        elif self.subject_span == s2 and self.object_span == s1:
            return (s2, s1)
        return None


def parse(text: str) -> Doc:
    """Parse text into dependency structure."""
    return nlp(text)


def convert_entity(entity: dict[str, Any], doc: Doc) -> Span | None:
    """Map character interval to spaCy span."""
    start_char = entity["char_interval"]["start_pos"]
    end_char = entity["char_interval"]["end_pos"]

    indices = []
    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)

        # check overlap
        if not (token_end <= start_char or token_start >= end_char):
            indices.append(token.i)

    if not indices:
        return None

    start_idx = min(indices)
    end_idx = max(indices) + 1

    return doc[start_idx:end_idx]


def find_lca(t1: Token, t2: Token) -> Token | None:
    """Find lowest common ancestor in dependency tree."""
    ancestors1 = set([t1] + list(t1.ancestors))
    ancestors2 = set([t2] + list(t2.ancestors))

    common = ancestors1 & ancestors2
    if not common:
        return None

    # return ancestor with minimum depth
    return min(common, key=lambda t: len(list(t.ancestors)))


def collect_conjuncts(token: Token) -> list[Token]:
    """Recursively collect coordinated tokens via 'conj' dependency."""
    result = [token]
    for child in token.children:
        if child.dep_ == "conj":
            result.extend(collect_conjuncts(child))
    return result


def check_negation(verb: Token) -> bool:
    """Detect negation marker."""
    return any(child.dep_ == "neg" for child in verb.children)


def detect_modality(verb: Token) -> str | None:
    """Extract modal auxiliary."""
    modals = {"may", "might", "could", "should", "would", "must", "can"}

    for child in verb.children:
        if child.dep_ == "aux" and child.pos_ == "VERB":
            if child.lemma_ in modals:
                return child.lemma_

    return None


def collect_path_verbs(r1: Token, r2: Token, lca: Token) -> list[Token]:
    """Collect verbs on paths from entity roots to LCA."""
    verbs = []

    # traverse from r1 -> lca
    current = r1
    while current and current != lca:
        if current.pos_ == "VERB":
            verbs.append(current)
        current = current.head

    # traverse from r2 -> lca
    current = r2
    while current and current != lca:
        if current.pos_ == "VERB":
            verbs.append(current)
        current = current.head

    return verbs


def find_syntactic_head(entity_span: Span, candidate_token: Token) -> bool:
    """Check if a token is the syntactic head of a (larger) entity span.

    Important for linking a full entity span
    (e.g., 'the new ML model') to its syntactic argument
    (e.g., the token 'model').

    Returns true if:
        1. The token is *inside* the span.
        2. The token is the head of a word *inside* the span (transitive).
    """
    # direct containment
    if candidate_token in entity_span:
        return True

    # transitive headship: entity modifies the candidate
    for entity_token in entity_span:
        current = entity_token
        # traverse up dependency tree until reaching root
        while current.head != current:
            if current.head == candidate_token:
                return True
            current = current.head

    return False


def analyze_verb(verb: Token, s1: Span, s2: Span) -> list[VerbRelation]:
    """Check if verb links s1 and s2 as its subject and object.

    Relation-finding logic. It handles:
    - Passive voice (e.g., "model was trained by Google")
    - Coordination (e.g., "Google and Meta trained...")
    - Mapping spans to roles via `find_syntactic_head`
    """
    children = {child: child.dep_ for child in verb.children}
    has_passive = any(dep == "nsubjpass" for dep in children.values())

    # extract metadata ONCE
    is_negated = check_negation(verb)
    modality = detect_modality(verb)

    agent_tokens = []
    patient_tokens = []

    if has_passive:
        # passive: patient is syntactic subject
        for child, dep_rel in children.items():
            if dep_rel == "nsubjpass":
                patient_tokens.extend(collect_conjuncts(child))

        # agent in 'by' phrase
        for child in verb.children:
            if child.dep_ == "agent" and child.text.lower() == "by":
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        agent_tokens.extend(collect_conjuncts(prep_child))
    else:
        # active: direct mapping
        for child, dep_rel in children.items():
            if dep_rel in ("nsubj", "csubj"):
                agent_tokens.extend(collect_conjuncts(child))
            elif dep_rel in ("dobj", "attr"):
                patient_tokens.extend(collect_conjuncts(child))
            elif dep_rel == "pobj":
                patient_tokens.extend(collect_conjuncts(child))

    # generate relations for all agent-patient combinations
    relations = []
    for agent_tok in agent_tokens:
        for patient_tok in patient_tokens:
            agent_span = None
            patient_span = None

            # apply transitive head detection
            if find_syntactic_head(s1, agent_tok):
                agent_span = s1
            elif find_syntactic_head(s2, agent_tok):
                agent_span = s2

            if find_syntactic_head(s1, patient_tok):
                patient_span = s1
            elif find_syntactic_head(s2, patient_tok):
                patient_span = s2

            if agent_span and patient_span and agent_span != patient_span:
                relations.append(
                    VerbRelation(
                        verb=verb.lemma_,
                        subject_span=agent_span,
                        object_span=patient_span,
                        negated=is_negated,
                        modal=modality,
                    )
                )

    return relations


def find_relations(s1: Span, s2: Span) -> list[VerbRelation]:
    """Find all verb-mediated relations between entity spans."""
    r1 = s1.root
    r2 = s2.root

    lca = find_lca(r1, r2)
    if not lca:
        return []

    relations = []

    # check if lca is connecting verb
    if lca.pos_ == "VERB":
        relations.extend(analyze_verb(lca, s1, s2))

    # check verbs on path to lca
    path_verbs = collect_path_verbs(r1, r2, lca)
    for verb_token in path_verbs:
        relations.extend(analyze_verb(verb_token, s1, s2))

    return relations
