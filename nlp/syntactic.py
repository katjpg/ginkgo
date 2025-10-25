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
            print(f"  {token.text:15} --{token.dep_:10}--> {token.head.text:15} (POS: {token.pos_})")
        print()


def extract_deps(doc: Doc) -> list[dict]:
    dependency_info = []
    
    for sent in doc.sents:
        sent_info = {
            'text': sent.text,
            'tokens': []
        }
        
        for token in sent:
            token_info = {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            }
            sent_info['tokens'].append(token_info)
        
        dependency_info.append(sent_info)
    
    return dependency_info


def convert_entity(entity: dict, doc: Doc) -> Span | None:
    start_char = entity["char_interval"]["start_pos"]
    end_char = entity["char_interval"]["end_pos"]
    
    indices = []
    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        if not (token_end <= start_char or token_start >= end_char):
            indices.append(token.i)
    
    if not indices:
        return None
    
    start_idx = min(indices)
    end_idx = max(indices) + 1
    
    return doc[start_idx:end_idx]


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
