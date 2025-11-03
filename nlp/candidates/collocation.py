from dataclasses import dataclass
from typing import Any


@dataclass
class CollocationCandidate:
    """Candidate pair with collocation scores."""
    e1: str
    e2: str
    npmi: float
    n_cooc: int
    type_compatible: bool
    sent_distance: int


class Collocation:
    """Collocation-based candidate filter."""
    
    def __init__(self, 
                 min_cooc: int = 1,
                 max_sent_dist: int = 5,
                 allowed_types: set[tuple[str, str]] | None = None):
        """Initialize with thresholds and allowed type pairs."""
        self.min_cooc = min_cooc
        self.max_sent_dist = max_sent_dist
        
        self.allowed_types = allowed_types or {
            ("method", "task"),
            ("method", "dataset"),
            ("method", "method"),
            ("method", "metric"),
            ("task", "dataset"),
            ("task", "metric"),
            ("dataset", "metric"),
        }
    
    def filter(self, 
               candidates: list[Any],
               entity_dict: dict[str, dict]) -> list[CollocationCandidate]:
        """Filter candidates using collocation constraints."""
        filtered = []
        
        for c in candidates:
            e1_type = entity_dict.get(c.e1, {}).get('type', 'other')
            e2_type = entity_dict.get(c.e2, {}).get('type', 'other')
            
            type_compat = self._check_types(e1_type, e2_type)
            
            e1_sent_idx = entity_dict.get(c.e1, {}).get('sentence_index', 0)
            e2_sent_idx = entity_dict.get(c.e2, {}).get('sentence_index', 0)
            sent_dist = abs(e1_sent_idx - e2_sent_idx)
            
            cooc_pass = c.n_cooc >= self.min_cooc
            dist_pass = sent_dist <= self.max_sent_dist
            
            if type_compat and cooc_pass and dist_pass:
                filtered.append(CollocationCandidate(
                    e1=c.e1,
                    e2=c.e2,
                    npmi=c.npmi,
                    n_cooc=c.n_cooc,
                    type_compatible=type_compat,
                    sent_distance=sent_dist
                ))
        
        return sorted(filtered, key=lambda x: x.npmi, reverse=True)
    
    def _check_types(self, t1: str, t2: str) -> bool:
        """Check if type pair is allowed."""
        pair = tuple(sorted([t1, t2]))
        return pair in self.allowed_types or (t1, t2) in self.allowed_types
    
    def stats(self, candidates: list[CollocationCandidate]) -> dict[str, float | int]:
        """Compute summary statistics."""
        if not candidates:
            return {
                'n_pairs': 0,
                'type_pass_rate': 0.0,
                'avg_sent_dist': 0.0,
                'avg_cooc': 0.0,
                'min_cooc': self.min_cooc,
                'max_sent_dist': self.max_sent_dist
            }
        
        n = len(candidates)
        type_pass = sum(c.type_compatible for c in candidates)
        avg_dist = sum(c.sent_distance for c in candidates) / n
        avg_cooc = sum(c.n_cooc for c in candidates) / n
        
        return {
            'n_pairs': n,
            'type_pass_rate': type_pass / n if n > 0 else 0.0,
            'avg_sent_dist': avg_dist,
            'avg_cooc': avg_cooc,
            'min_cooc': self.min_cooc,
            'max_sent_dist': self.max_sent_dist
        }



""" 
# quick demonstration

if __name__ == "__main__":
    from nlp.candidates.npmi import NPMI, EntityPair
    
    entities = [
        {'text': 'BERT', 'type': 'method', 'sentence_index': 2},
        {'text': 'attention mechanism', 'type': 'other', 'sentence_index': 2},
        {'text': 'SQuAD', 'type': 'dataset', 'sentence_index': 5},
        {'text': 'F1 score', 'type': 'metric', 'sentence_index': 5},
    ]
    
    entity_dict = {e['text']: e for e in entities}
    
    npmi_candidates = [
        EntityPair('BERT', 'attention mechanism', 0.58, 3, 10, 8),
        EntityPair('BERT', 'SQuAD', 0.42, 2, 10, 5),
        EntityPair('SQuAD', 'F1 score', 0.39, 4, 5, 7),
        EntityPair('attention mechanism', 'F1 score', 0.15, 1, 8, 7),
    ]
    
    collocation = Collocation(min_cooc=1, max_sent_dist=5)
    
    filtered = collocation.filter(npmi_candidates, entity_dict)
    
    print(f"Filtered {len(filtered)} candidates:\n")
    for c in filtered:
        print(f"{c.e1:25s} ↔ {c.e2:25s}")
        print(f"  NPMI: {c.npmi:.3f} | Co-occ: {c.n_cooc} | Sent dist: {c.sent_distance}")
        print(f"  Type compatible: {c.type_compatible}\n")
    
    stats = collocation.stats(filtered)
    print("Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

"""