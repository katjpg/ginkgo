import spacy
from spacy.tokens import Doc, Span, Token
from dataclasses import dataclass
import numpy as np

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


@dataclass
class EntityCentricTree:
    """Entity-centric dependency tree for relation extraction."""
    entity_span: Span
    edges: dict[Token, int]
    sentence: Span | None
    
    def to_adjacency_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Convert to adjacency matrix for GNN processing."""
        if not self.sentence:
            return np.array([]), []
        
        all_tokens = list(self.sentence)
        n = len(all_tokens)
        matrix = np.zeros((n, n), dtype=np.int32)
        
        entity_indices = {t.i for t in self.entity_span}
        
        for i, token_i in enumerate(all_tokens):
            for j, token_j in enumerate(all_tokens):
                if i == j:
                    matrix[i][j] = 0
                elif token_i.i in entity_indices and token_j.i not in entity_indices:
                    matrix[i][j] = self.edges.get(token_j, 999)
                elif token_i.i not in entity_indices and token_j.i in entity_indices:
                    if token_i in self.edges:
                        matrix[i][j] = self.edges[token_i]
        
        token_texts = [t.text for t in all_tokens]
        return matrix, token_texts
    
    def get_features(self) -> dict:
        """Extract features for downstream processing."""
        return {
            'entity_text': self.entity_span.text,
            'entity_root': self.entity_span.root.text,
            'entity_pos': [t.pos_ for t in self.entity_span],
            'connected_tokens': len(self.edges),
            'max_distance': max(self.edges.values()) if self.edges else 0,
            'avg_distance': np.mean(list(self.edges.values())) if self.edges else 0
        }


def calculate_syntactic_distance(token1: Token, token2: Token) -> int:
    """Calculate syntactic distance between tokens in dependency tree."""
    if token1 == token2:
        return 0
    
    path1 = []
    current = token1
    while current.head != current:
        path1.append(current)
        current = current.head
    path1.append(current)
    
    path2 = []
    current = token2
    while current.head != current:
        path2.append(current)
        current = current.head
    path2.append(current)
    
    path1_set = set(path1)
    for i, token in enumerate(path2):
        if token in path1_set:
            lca_idx1 = path1.index(token)
            return lca_idx1 + i
    
    return 999


def build_entity_centric_tree(doc: Doc, entity_span: Span) -> EntityCentricTree:
    """Build entity-centric dependency tree (Algorithm 1)."""
    edges = {}
    
    sent = None
    for s in doc.sents:
        if entity_span.start >= s.start and entity_span.end <= s.end:
            sent = s
            break
    
    if not sent:
        return EntityCentricTree(entity_span=entity_span, edges={}, sentence=None)
    
    for token in sent:
        if token.i >= entity_span.start and token.i < entity_span.end:
            continue
        
        min_dist = 999
        for entity_token in entity_span:
            dist = calculate_syntactic_distance(token, entity_token)
            if dist >= 0:
                min_dist = min(min_dist, dist)
        
        if min_dist != 999:
            edges[token] = min_dist
    
    return EntityCentricTree(entity_span=entity_span, edges=edges, sentence=sent)


def extract_entity_pair_features(doc: Doc, e1: Span, e2: Span) -> dict:
    """Extract features from entity-centric trees for relation extraction."""
    tree1 = build_entity_centric_tree(doc, e1)
    tree2 = build_entity_centric_tree(doc, e2)
    
    e1_to_e2_dist = min(
        calculate_syntactic_distance(t1, t2)
        for t1 in e1 for t2 in e2
    )
    
    path_tokens = find_sdp(e1.root, e2.root)
    
    return {
        'tree1_features': tree1.get_features(),
        'tree2_features': tree2.get_features(),
        'entity_distance': e1_to_e2_dist,
        'path_length': len(path_tokens),
        'path_has_verb': any(t.pos_ == 'VERB' for t in path_tokens),
        'path_verbs': [t.lemma_ for t in path_tokens if t.pos_ == 'VERB']
    }


def format_for_relation_prompt(doc: Doc, e1: Span, e2: Span) -> str:
    """Format entity-centric features for relation classification."""
    features = extract_entity_pair_features(doc, e1, e2)
    
    tree1 = build_entity_centric_tree(doc, e1)
    tree2 = build_entity_centric_tree(doc, e2)
    
    e1_context = [t.text for t, d in tree1.edges.items() if d <= 2][:5]
    e2_context = [t.text for t, d in tree2.edges.items() if d <= 2][:5]
    
    parts = []
    
    if features['path_verbs']:
        verbs = ', '.join(f"'{v}'" for v in features['path_verbs'])
        parts.append(f"via {verbs}")
    
    if e1_context:
        ctx = ' '.join(e1_context)
        parts.append(f"[{e1.text}: {ctx}]")
    
    if e2_context:
        ctx = ' '.join(e2_context)
        parts.append(f"[{e2.text}: {ctx}]")
    
    return '; '.join(parts) if parts else 'direct connection'


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
    """Find entity by text matching within sentence boundaries."""
    if "text" not in entity:
        return None
    
    entity_lower = entity["text"].lower()
    
    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        if entity_lower not in sent_text_lower:
            continue
            
        for i in range(len(sent)):
            for j in range(i + 1, min(i + 10, len(sent) + 1)):
                span = doc[sent.start + i:sent.start + j]
                if span.text.lower() == entity_lower:
                    return span
    
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
    """Extract shortest dependency path between tokens."""
    lca = find_lca(t1, t2)
    if not lca:
        return []
    
    up_path = []
    curr = t1
    while curr != lca:
        up_path.append(curr)
        curr = curr.head
    
    down_path = []
    curr = t2
    while curr != lca:
        down_path.append(curr)
        curr = curr.head
    
    path = up_path + [lca] + list(reversed(down_path))
    
    seen = set()
    unique_path = []
    for token in path:
        if token.i not in seen:
            unique_path.append(token)
            seen.add(token.i)
    
    return unique_path


def get_1hop(sdp: list[Token]) -> list[Token]:
    """Get 1-hop neighbors from shortest dependency path."""
    neighbors = []
    sdp_set = set(sdp)
    
    important_deps = {"neg", "amod", "advmod", "compound", "nummod"}
    
    for token in sdp:
        for child in token.children:
            if child not in sdp_set and child.dep_ in important_deps:
                neighbors.append(child)
    
    return neighbors


def extract_path(t1: Token, t2: Token) -> dict:
    """Extract structured SDP features for entity pair."""
    sdp = find_sdp(t1, t2)
    
    if not sdp:
        return {"sdp": [], "context": [], "verb": None, "preps": [], "negated": False}
    
    context = get_1hop(sdp)
    sdp_texts = [t.text for t in sdp]
    context_texts = [t.text for t in context]
    
    verb = None
    for token in sdp:
        if token.pos_ == "VERB":
            verb = token.text
            break
    
    preps = [t.text for t in sdp if t.pos_ == "ADP"]
    negated = any(t.dep_ == "neg" for t in context)
    
    return {
        "sdp": sdp_texts,
        "context": context_texts,
        "verb": verb,
        "preps": preps,
        "negated": negated,
    }


def verbalize_path(t1: Token, t2: Token) -> str:
    """Convert dependency path to natural language summary."""
    features = extract_path(t1, t2)
    
    if not features["sdp"]:
        return "no connection"
    
    parts = []
    
    if features["verb"]:
        parts.append(f"via '{features['verb']}'")
    
    if features["preps"]:
        preps_str = ", ".join(f"'{p}'" for p in features["preps"])
        parts.append(f"using {preps_str}")
    
    if features["negated"]:
        parts.append("(negated)")
    
    if parts:
        return "; ".join(parts)
    else:
        return "direct connection"
    
def extract_exemplar(doc: Doc, e1: str, e2: str) -> str | None:
    """Extract exemplar containing both entities."""
    e1_lower = e1.lower()
    e2_lower = e2.lower()
    
    for sent in doc.sents:
        sent_text = sent.text
        if e1_lower not in sent_text.lower():
            continue
        if e2_lower not in sent_text.lower():
            continue
        
        tokens = list(sent)
        
        if len(tokens) <= 30:
            return sent_text
        
        e1_idx = -1
        e2_idx = -1
        
        for i, token in enumerate(tokens):
            if token.text.lower() == e1_lower:
                e1_idx = i
            if token.text.lower() == e2_lower:
                e2_idx = i
        
        if e1_idx == -1 or e2_idx == -1:
            return sent_text
        
        start = max(0, min(e1_idx, e2_idx) - 5)
        end = min(len(tokens), max(e1_idx, e2_idx) + 6)
        
        return " ".join(t.text for t in tokens[start:end])
    
    return None




def mask_entities(text: str, e1: str, e2: str) -> str | None:
    """Replace entity mentions with [ENT1] and [ENT2]."""
    
    if not text or not e1 or not e2:
        return None
    
    text_lower = text.lower()
    e1_lower = e1.lower()
    e2_lower = e2.lower()
    
    if e1_lower not in text_lower or e2_lower not in text_lower:
        return None
    
    idx1 = text_lower.find(e1_lower)
    idx2 = text_lower.find(e2_lower)
    
    if idx1 == -1 or idx2 == -1:
        return None
    
    if idx1 < idx2:
        before = text[:idx1]
        after = text[idx1 + len(e1):]
        masked = before + "[ENT1]" + after
        
        idx2_adj = idx2 - (len(e1) - len("[ENT1]"))
        e2_end = idx2 + len(e2)
        before2 = masked[:idx2_adj]
        after2 = masked[e2_end - (len(e1) - len("[ENT1]")):]
        masked = before2 + "[ENT2]" + after2
    else:
        before = text[:idx2]
        after = text[idx2 + len(e2):]
        masked = before + "[ENT2]" + after
        
        idx1_adj = idx1 - (len(e2) - len("[ENT2]"))
        e1_end = idx1 + len(e1)
        before1 = masked[:idx1_adj]
        after1 = masked[e1_end - (len(e2) - len("[ENT2]")):]
        masked = before1 + "[ENT1]" + after1
    
    return masked


def find_entity_span(doc: Doc, entity_text: str) -> Span | None:
    """Find entity span in doc via substring matching."""
    
    if not entity_text:
        return None
    
    entity_lower = entity_text.lower()
    
    for token in doc:
        if token.text.lower() == entity_lower:
            return doc[token.i:token.i + 1]
    
    for i in range(len(doc)):
        candidate = doc[i:min(i + 10, len(doc))]
        if candidate.text.lower() == entity_lower:
            return candidate
    
    doc_lower = doc.text.lower()
    if entity_lower in doc_lower:
        idx = doc_lower.find(entity_lower)
        start_token = None
        for token in doc:
            if token.idx <= idx < token.idx + len(token.text):
                start_token = token.i
                break
        
        if start_token is not None:
            end_char = idx + len(entity_text)
            for token in doc[start_token:]:
                if token.idx + len(token.text) >= end_char:
                    return doc[start_token:token.i + 1]
    
    return None
