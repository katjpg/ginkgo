import numpy as np
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ImplicitRelation:
    """Inferred implicit relation with bridge entities."""
    e_i: str
    e_j: str
    confidence: float
    bridges: list[str]
    n_bridges: int


class ImplicitCluster:
    """Discover implicit relations through k-NN neighborhoods."""
    
    def __init__(
        self,
        k: int = 5,
        tau_sim: float = 0.70,
        tau_b: int = 2
    ):
        """Initialize with neighborhood and bridge thresholds."""
        self.k = k
        self.tau_sim = tau_sim
        self.tau_b = tau_b
    
    def infer(
        self,
        entities: list[dict],
        embeddings: np.ndarray,
        explicit_pairs: set[tuple[str, str]]
    ) -> list[ImplicitRelation]:
        """Infer implicit relations from entity embeddings."""
        entity_texts = [e['text'] for e in entities]
        
        if len(entity_texts) < 2:
            return []
        
        neighborhoods = self._build_neighborhoods(entity_texts, embeddings)
        
        implicit_rels = self._find_common_neighbors(
            entity_texts,
            neighborhoods,
            explicit_pairs
        )
        
        return sorted(implicit_rels, key=lambda x: x.confidence, reverse=True)
    
    def _build_neighborhoods(
        self,
        entity_texts: list[str],
        embeddings: np.ndarray
    ) -> dict[str, list[str]]:
        """Build k-NN neighborhoods for all entities."""
        neighborhoods = {}
        
        if len(entity_texts) == 1:
            neighborhoods[entity_texts[0]] = []
            return neighborhoods
        
        sim_matrix = cosine_similarity(embeddings)
        
        for i, e_i in enumerate(entity_texts):
            sims = sim_matrix[i]
            
            # exclude self (set to -inf for sorting)
            sims_copy = sims.copy()
            sims_copy[i] = -np.inf
            
            # get top k indices
            sorted_idx = np.argsort(sims_copy)[::-1][:self.k]
            
            # filter by similarity threshold
            neighbors = [
                entity_texts[j] for j in sorted_idx
                if sims[j] >= self.tau_sim
            ]
            
            neighborhoods[e_i] = neighbors
        
        return neighborhoods
    
    def _find_common_neighbors(
        self,
        entity_texts: list[str],
        neighborhoods: dict[str, list[str]],
        explicit_pairs: set[tuple[str, str]]
    ) -> list[ImplicitRelation]:
        """Find implicit relations via common neighbor analysis."""
        implicit_rels = []
        
        for i, e_i in enumerate(entity_texts):
            for j in range(i + 1, len(entity_texts)):
                e_j = entity_texts[j]
                
                # check if pair already explicit
                pair = tuple(sorted([e_i, e_j]))
                if pair in explicit_pairs:
                    continue
                
                # skip if direct similarity too high (likely explicit)
                n_i = set(neighborhoods.get(e_i, []))
                n_j = set(neighborhoods.get(e_j, []))
                
                # check direct membership (high similarity)
                if e_j in n_i or e_i in n_j:
                    continue
                
                # find bridge entities
                bridges = n_i & n_j
                
                if len(bridges) >= self.tau_b:
                    confidence = len(bridges) / self.k if self.k > 0 else 0.0
                    implicit_rels.append(ImplicitRelation(
                        e_i=e_i,
                        e_j=e_j,
                        confidence=min(confidence, 1.0),
                        bridges=sorted(list(bridges)),
                        n_bridges=len(bridges)
                    ))
        
        return implicit_rels
    
    def stats(self, implicit_rels: list[ImplicitRelation]) -> dict:
        """Compute summary statistics."""
        if not implicit_rels:
            return {
                'n_implicit': 0,
                'mean_confidence': 0.0,
                'median_confidence': 0.0,
                'mean_bridges': 0.0,
                'k': self.k,
                'tau_b': self.tau_b
            }
        
        confidences = np.array([r.confidence for r in implicit_rels])
        n_bridges = np.array([r.n_bridges for r in implicit_rels])
        
        return {
            'n_implicit': len(implicit_rels),
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'mean_bridges': float(np.mean(n_bridges)),
            'max_confidence': float(np.max(confidences)),
            'min_confidence': float(np.min(confidences)),
            'k': self.k,
            'tau_b': self.tau_b
        }