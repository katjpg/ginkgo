import spacy
from spacy.tokens import Doc, Span, Token
from dataclasses import dataclass

nlp = spacy.load("en_core_web_sm")


@dataclass
class VerbRelation:
    verb: str
    subject_span: Span
    object_span: Span
    negated: bool
    modal: str | None

    def matches(self, s1: Span, s2: Span) -> tuple[Span, Span] | None:
        if self.subject_span == s1 and self.object_span == s2:
            return (s1, s2)
        elif self.subject_span == s2 and self.object_span == s1:
            return (s2, s1)
        return None


def parse(text: str) -> Doc:
    return nlp(text)


def print_parse(doc: Doc) -> None:
    print("=== DEPENDENCY PARSE ===\n")

    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        for token in sent:
            print(
                f"  {token.text:15} --{token.dep_:10}--> {token.head.text:15} (POS: {token.pos_})"
            )
        print()


def extract_deps(doc: Doc) -> list[dict]:
    dependency_info = []

    for sent in doc.sents:
        sent_info = {"text": sent.text, "tokens": []}

        for token in sent:
            token_info = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.text,
                "head_pos": token.head.pos_,
                "children": [child.text for child in token.children],
            }
            sent_info["tokens"].append(token_info)

        dependency_info.append(sent_info)

    return dependency_info


def convert_entity(entity: dict, doc: Doc) -> Span | None:
    """Find entity by text matching instead of character positions."""
    if "text" not in entity:
        return None
    
    entity_lower = entity["text"].lower()
    
    # 1. exact match first
    for sent in doc.sents:
        for i in range(len(sent)):
            for j in range(i + 1, min(i + 10, len(sent) + 1)):
                span = sent[i:j]
                if span.text.lower() == entity_lower:
                    return span
    
    # 2. partial match
    for sent in doc.sents:
        for token in sent:
            if entity_lower in token.text.lower() or token.text.lower() in entity_lower:
                return doc[token.i:token.i+1]
    
    return None


def find_lca(t1: Token, t2: Token) -> Token | None:
    ancestors1 = set([t1] + list(t1.ancestors))
    ancestors2 = set([t2] + list(t2.ancestors))

    common = ancestors1 & ancestors2
    if not common:
        return None

    return min(common, key=lambda t: len(list(t.ancestors)))


def collect_conjuncts(token: Token) -> list[Token]:
    result = [token]
    for child in token.children:
        if child.dep_ == "conj":
            result.extend(collect_conjuncts(child))
    return result


def check_negation(verb: Token) -> bool:
    return any(child.dep_ == "neg" for child in verb.children)


def detect_modality(verb: Token) -> str | None:
    modals = {"may", "might", "could", "should", "would", "must", "can"}

    for child in verb.children:
        if child.dep_ == "aux" and child.pos_ == "VERB":
            if child.lemma_ in modals:
                return child.lemma_

    return None


def collect_verbs(r1: Token, r2: Token, lca: Token) -> list[Token]:
    verbs = []

    current = r1
    while current and current != lca:
        if current.pos_ == "VERB":
            verbs.append(current)
        current = current.head

    current = r2
    while current and current != lca:
        if current.pos_ == "VERB":
            verbs.append(current)
        current = current.head

    return verbs


def find_head(entity_span: Span, candidate_token: Token) -> bool:
    if candidate_token in entity_span:
        return True

    for entity_token in entity_span:
        current = entity_token
        while current.head != current:
            if current.head == candidate_token:
                return True
            current = current.head

    return False


def analyze_verb(verb: Token, s1: Span, s2: Span) -> list[VerbRelation]:
    children = {child: child.dep_ for child in verb.children}
    has_passive = any(dep == "nsubjpass" for dep in children.values())

    is_negated = check_negation(verb)
    modality = detect_modality(verb)

    agent_tokens = []
    patient_tokens = []

    if has_passive:
        for child, dep_rel in children.items():
            if dep_rel == "nsubjpass":
                patient_tokens.extend(collect_conjuncts(child))

        for child in verb.children:
            if child.dep_ == "agent" and child.text.lower() == "by":
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        agent_tokens.extend(collect_conjuncts(prep_child))
    else:
        for child, dep_rel in children.items():
            if dep_rel in ("nsubj", "csubj"):
                agent_tokens.extend(collect_conjuncts(child))
            elif dep_rel in ("dobj", "attr"):
                patient_tokens.extend(collect_conjuncts(child))
            elif dep_rel == "pobj":
                patient_tokens.extend(collect_conjuncts(child))

    relations = []
    for agent_tok in agent_tokens:
        for patient_tok in patient_tokens:
            agent_span = None
            patient_span = None

            if find_head(s1, agent_tok):
                agent_span = s1
            elif find_head(s2, agent_tok):
                agent_span = s2

            if find_head(s1, patient_tok):
                patient_span = s1
            elif find_head(s2, patient_tok):
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
    r1 = s1.root
    r2 = s2.root

    lca = find_lca(r1, r2)
    if not lca:
        return []

    relations = []

    if lca.pos_ == "VERB":
        relations.extend(analyze_verb(lca, s1, s2))

    path_verbs = collect_verbs(r1, r2, lca)
    for verb_token in path_verbs:
        relations.extend(analyze_verb(verb_token, s1, s2))

    return relations


def find_sdp(t1: Token, t2: Token) -> list[Token]:
    """Extract shortest dependency path between tokens.

    Finds the minimal grammatical path connecting two entities by traversing
    up from each token to their lca, then combining the paths.

    Args:
        t1: First token (typically entity root)
        t2: Second token (typically entity root)

    Returns:
        List of tokens forming the shortest dependency path, empty if no path exists
    """
    lca = find_lca(t1, t2)
    if not lca:
        return []

    path = []

    # traverse from first token up to LCA
    curr = t1
    while curr != lca:
        path.append(curr)
        curr = curr.head

    # add the LCA itself
    path.append(lca)

    # collect path from second token to LCA (will reverse later)
    down_path = []
    curr = t2
    while curr != lca:
        down_path.append(curr)
        curr = curr.head

    # add reversed path from LCA to second token
    path.extend(reversed(down_path))

    return path


def get_1hop(sdp: list[Token]) -> list[Token]:
    """Get 1-hop neighbors from shortest dependency path.

    Identifies tokens exactly one dependency edge away from any SDP token.

    Focuses on modifiers (negation, adjectives, compounds) that specify
    entity meaning + eliminates noise.

    Captures important context like "linear" in "linear regression" or "speaker" in
    "speaker verification".

    Args:
        sdp: List of tokens forming the shortest dependency path

    Returns:
        List of 1-hop neighbor tokens with semantically important dependency types
    """
    neighbors = []
    sdp_set = set(sdp)

    # dependency types that typically carry important semantic information
    important_deps = {
        "neg",  # negation (critical for meaning)
        "amod",  # adjectival modifier (e.g., "linear" regression)
        "advmod",  # adverbial modifier (e.g., "significantly" improves)
        "compound",  # compound modifier (e.g., "speaker" verification)
        "nummod",  # numeric modifier (e.g., "three" methods)
    }

    for token in sdp:
        for child in token.children:
            # only include children not already in SDP and with important dependency types
            if child not in sdp_set and child.dep_ in important_deps:
                neighbors.append(child)

    return neighbors


def extract_path(t1: Token, t2: Token) -> dict:
    """Extract structured SDP features for entity pair.

    Transforms raw dependency parsing into machine-readable features that capture
    syntactic relationships.

    Structures information about verbs, prepositions,
    and negation for downstream relation extraction processing.

    Args:
        t1: First entity's root token
        t2: Second entity's root token

    Returns:
        Dictionary containing:
        - sdp: List of token texts in shortest path
        - context: List of 1-hop neighbor token texts
        - verb: First verb in path (if any)
        - preps: All prepositions in path
        - negated: Boolean indicating negation presence
    """
    sdp = find_sdp(t1, t2)

    # handle case where no path exists (e.g., disconnected parse)
    if not sdp:
        return {"sdp": [], "context": [], "verb": None, "preps": [], "negated": False}

    # get contextual modifiers
    context = get_1hop(sdp)

    # extract linguistic features from path
    sdp_texts = [t.text for t in sdp]
    context_texts = [t.text for t in context]

    # find first verb in path (often the main predicate)
    verb = None
    for token in sdp:
        if token.pos_ == "VERB":
            verb = token.text
            break

    # collect prepositions (indicate relationships like "of", "in", "for")
    preps = [t.text for t in sdp if t.pos_ == "ADP"]

    # check for negation in both path and context
    negated = any(t.dep_ == "neg" for t in context)

    return {
        "sdp": sdp_texts,
        "context": context_texts,
        "verb": verb,
        "preps": preps,
        "negated": negated,
    }


def verbalize_path(t1: Token, t2: Token) -> str:
    """Convert dependency path to natural language summary.

    Transforms tree structures into interpretable descriptions that LLMs can
    understand.

    Args:
        t1: First entity's root token
        t2: Second entity's root token

    Returns:
        Natural language description of the syntactic relationship,
        e.g., "via 'improves' using 'on'; (negated)" or "no connection"
    """
    features = extract_path(t1, t2)

    # handle disconnected entities
    if not features["sdp"]:
        return "no connection"

    parts = []

    # describe verb-mediated relationship
    if features["verb"]:
        parts.append(f"via '{features['verb']}'")

    # describe prepositional relationships
    if features["preps"]:
        preps_str = ", ".join(f"'{p}'" for p in features["preps"])
        parts.append(f"using {preps_str}")

    # flag negation as it critically changes meaning
    if features["negated"]:
        parts.append("(negated)")

    # combine parts or indicate direct connection
    if parts:
        return "; ".join(parts)
    else:
        return "direct connection"
