import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class EntityPair:
    """Entity pair with NPMI score and occurrence statistics."""
    e1: str
    e2: str
    npmi: float
    n_cooc: int
    n_e1: int
    n_e2: int


class NPMI:
    """NPMI-based candidate pair selector."""
    
    def __init__(self, tau: float = 0.15, min_cooc: int = 1):
        """Initialize with threshold tau and minimum co-occurrence."""
        self.tau = tau
        self.min_cooc = min_cooc
    
    def select(self, entities: list[dict[str, Any]], sentences: list[str]) -> list[EntityPair]:
        """Select candidate pairs using NPMI scoring."""
        entity_texts = [e['text'] for e in entities]
        
        counts_e, counts_ij = self._count(entity_texts, sentences)
        
        candidates = self._score(entity_texts, counts_e, counts_ij, len(sentences))
        
        candidates = [
            c for c in candidates 
            if c.npmi >= self.tau and c.n_cooc >= self.min_cooc
        ]
        
        return sorted(candidates, key=lambda x: x.npmi, reverse=True)
    
    def _count(self, 
               entity_texts: list[str], 
               sentences: list[str]) -> tuple[Counter, Counter]:
        """Count entity and co-occurrence frequencies."""
        counts_e = Counter()
        counts_ij = Counter()
        
        for s in sentences:
            s_lower = s.lower()
            entities_in_s = [e for e in entity_texts if e.lower() in s_lower]
            
            for e in entities_in_s:
                counts_e[e] += 1
            
            for i, e1 in enumerate(entities_in_s):
                for e2 in entities_in_s[i+1:]:
                    pair = tuple(sorted([e1, e2]))
                    counts_ij[pair] += 1
        
        return counts_e, counts_ij
    
    def _score(self,
               entity_texts: list[str],
               counts_e: Counter,
               counts_ij: Counter,
               N: int) -> list[EntityPair]:
        """Compute NPMI scores for all pairs."""
        candidates = []
        
        for (e1, e2), n_ij in counts_ij.items():
            if e1 not in counts_e or e2 not in counts_e:
                continue
            
            n_i = counts_e[e1]
            n_j = counts_e[e2]
            
            p_i = n_i / N
            p_j = n_j / N
            p_ij = n_ij / N
            
            if p_ij == 0:
                continue
            
            pmi = np.log2(p_ij / (p_i * p_j))
            npmi = pmi / (-np.log2(p_ij))
            
            candidates.append(EntityPair(
                e1=e1,
                e2=e2,
                npmi=npmi,
                n_cooc=n_ij,
                n_e1=n_i,
                n_e2=n_j
            ))
        
        return candidates
    
    def stats(self, candidates: list[EntityPair]) -> dict[str, float | int]:
        """Compute summary statistics."""
        if not candidates:
            return {
                'n_pairs': 0,
                'mean': 0.0,
                'median': 0.0,
                'max': 0.0,
                'min': 0.0,
                'tau': self.tau
            }
        
        scores = np.array([c.npmi for c in candidates])
        
        return {
            'n_pairs': len(candidates),
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'max': float(np.max(scores)),
            'min': float(np.min(scores)),
            'tau': self.tau
        }



""" 

# quick demonstration

if __name__ == "__main__":
    entities = [
        {'text': 'BERT', 'type': 'method'},
        {'text': 'attention mechanism', 'type': 'other'},
        {'text': 'transformer', 'type': 'method'},
        {'text': 'masked language model', 'type': 'task'},
        {'text': 'SQuAD', 'type': 'dataset'},
    ]
    
    sentences = [
        "BERT uses attention mechanism for encoding.",
        "The transformer model relies on attention mechanism.",
        "BERT is pretrained with masked language model objective.",
        "We evaluate BERT on SQuAD dataset.",
        "Attention mechanism enables parallel computation.",
        "BERT achieves state-of-the-art on SQuAD.",
        "The masked language model helps BERT learn representations.",
        "Transformer architecture uses multi-head attention mechanism.",
    ]
    
    npmi = NPMI(tau=0.15, min_cooc=1)
    candidates = npmi.select(entities, sentences)
    
    print(f"Found {len(candidates)} candidates:\n")
    for c in candidates[:10]:
        print(f"{c.e1:30s} ↔ {c.e2:30s}")
        print(f"  NPMI: {c.npmi:.3f} | Co-occ: {c.n_cooc} | Freq: ({c.n_e1}, {c.n_e2})\n")
    
    stats = npmi.stats(candidates)
    print("Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
            


"""