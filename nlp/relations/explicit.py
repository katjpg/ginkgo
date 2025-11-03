"""
Workflow:
  1. Extract sentences containing both entities
  2. Mask entities to isolate relation patterns
  3. Encode with all-mpnet-base-v2 (768-dim)
  4. Cluster with AAP -> exemplars
  5. Label exemplars with Gemini via _label_relation
  6. Return (e1, rel_type, e2)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.llm import GeminiConfig
from llm.prompts.gemini import RELATION_PROMPT


@dataclass
class ExplicitRelation:
    """Discovered explicit relation with evidence."""
    e1: str
    e2: str
    rel_type: str
    confidence: float
    exemplar: str
    n_supporting: int


class Explicit:
    """Explicit relation discovery via AAP + Gemini."""
    
    def __init__(self,
                 gemini_config: GeminiConfig,
                 model_name: str = 'all-mpnet-base-v2',
                 damping: float = 0.9,
                 max_iter: int = 200,
                 conv_iter: int = 10) -> None:
        """Initialize with MPNet encoder + Gemini labeler."""
        self.model = SentenceTransformer(model_name)
        self.client = genai.Client(api_key=gemini_config.api_key)
        self.config = gemini_config
        self.damping = damping
        self.max_iter = max_iter
        self.conv_iter = conv_iter
    
    def discover(self,
                 candidates: List,
                 sentences: List[str],
                 entities: List[dict]) -> List[ExplicitRelation]:
        """Discover explicit relations from candidate pairs."""
        explicit_rels = []
        
        for cand in candidates:
            masked_sents = self._extract_masked_sentences(
                cand.e1, cand.e2, sentences
            )
            
            if len(masked_sents) < 1:
                continue
            
            if len(masked_sents) == 1:
                exemplar = masked_sents[0]
            else:
                embeddings = self.model.encode(masked_sents)
                exemplar_idx = self._aap_cluster(embeddings)
                exemplar = masked_sents[exemplar_idx]
            
            rel_type = self._label_relation(exemplar, cand.e1, cand.e2)
            confidence = min(len(masked_sents) / 10.0, 1.0)
            
            explicit_rels.append(ExplicitRelation(
                e1=cand.e1,
                e2=cand.e2,
                rel_type=rel_type,
                confidence=confidence,
                exemplar=exemplar,
                n_supporting=len(masked_sents)
            ))
        
        return sorted(explicit_rels, key=lambda x: x.confidence, reverse=True)
    
    def _extract_masked_sentences(self,
                                  e1: str,
                                  e2: str,
                                  sentences: List[str]) -> List[str]:
        """Extract sentences with both entities, replace with [ENT1]/[ENT2]."""
        masked = []
        e1_lower = e1.lower()
        e2_lower = e2.lower()
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            if e1_lower not in sent_lower or e2_lower not in sent_lower:
                continue
            
            masked_sent = sent.replace(e1, '[ENT1]', 1)
            masked_sent = masked_sent.replace(e2, '[ENT2]', 1)
            masked.append(masked_sent)
        
        return masked
    
    def _aap_cluster(self, embeddings: np.ndarray) -> int:
        """Run AAP clustering, return exemplar index."""
        m = len(embeddings)
        
        if m == 1:
            return 0
        
        S = cosine_similarity(embeddings)
        r = np.zeros((m, m), dtype=np.float32)
        a = np.zeros((m, m), dtype=np.float32)
        
        for iteration in range(self.max_iter):
            r_old = r.copy()
            a_old = a.copy()
            
            for i in range(m):
                for k in range(m):
                    mask = np.ones(m, dtype=bool)
                    mask[k] = False
                    max_val = np.max(a[i, mask] + S[i, mask])
                    r[i, k] = S[i, k] - max_val
            
            r = self.damping * r + (1 - self.damping) * r_old
            
            for i in range(m):
                for k in range(m):
                    if i == k:
                        r_mask = np.ones(m, dtype=bool)
                        r_mask[i] = False
                        a[i, k] = np.sum(np.maximum(0, r[r_mask, k]))
                    else:
                        mask = np.ones(m, dtype=bool)
                        mask[i] = False
                        mask[k] = False
                        sum_val = np.sum(np.maximum(0, r[mask, k]))
                        a[i, k] = np.minimum(0, r[k, k] + sum_val)
            
            a = self.damping * a + (1 - self.damping) * a_old
            
            exemplars_curr = np.array([np.argmax(a[i, :] + r[i, :]) for i in range(m)])
            exemplars_prev = np.array([np.argmax(a_old[i, :] + r_old[i, :]) for i in range(m)])
            
            if iteration >= self.conv_iter and np.array_equal(exemplars_curr, exemplars_prev):
                break
        
        exemplar_idx = int(np.argmax(np.diag(a + r)))
        return exemplar_idx
    
    def _label_relation(self, exemplar: str, e1: str, e2: str) -> str:
        """Label relation type using Gemini."""
        prompt_text = RELATION_PROMPT.format(exemplar=exemplar)
        
        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_output_tokens,
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt_text,
                config=generation_config,
            )
            
            if response.text is None:
                return 'related'
            
            rel_type = response.text.strip().lower()
            
            valid_types = {'uses', 'improves', 'evaluates', 'enables', 'proposes', 'related'}
            return rel_type if rel_type in valid_types else 'related'
        
        except Exception as e:
            print(f"Gemini API error: {e}. Falling back to 'related'")
            return 'related'
    
    def stats(self, relations: List[ExplicitRelation]) -> dict:
        """Compute summary statistics."""
        if not relations:
            return {
                'n_explicit': 0,
                'mean_confidence': 0.0,
                'median_confidence': 0.0,
                'mean_support': 0.0,
                'rel_type_dist': {}
            }
        
        confidences = np.array([r.confidence for r in relations])
        supports = np.array([r.n_supporting for r in relations])
        type_dist: dict[str, int] = {}
        
        for r in relations:
            type_dist[r.rel_type] = type_dist.get(r.rel_type, 0) + 1
        
        return {
            'n_explicit': len(relations),
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'mean_support': float(np.mean(supports)),
            'rel_type_dist': type_dist
        }


""" 

# quick demonstration

if __name__ == "__main__":
    config = GeminiConfig()
    
    candidates = [
        type('obj', (object,), {'e1': 'BERT', 'e2': 'attention'})(),
        type('obj', (object,), {'e1': 'transformer', 'e2': 'encoder'})(),
    ]
    
    sentences = [
        "BERT uses attention mechanism for encoding.",
        "The attention mechanism enables parallel computation.",
        "BERT leverages attention in transformers.",
        "The transformer uses encoder layers.",
        "Multi-head attention improves performance.",
        "Encoder and decoder use attention mechanisms.",
    ]
    
    explicit = Explicit(gemini_config=config)
    relations = explicit.discover(candidates, sentences, [])
    
    print(f"Found {len(relations)} explicit relations:\n")
    for rel in relations:
        print(f"{rel.e1:20s} --[{rel.rel_type:12s}]--> {rel.e2:20s}")
        print(f"  Confidence: {rel.confidence:.3f} | Supporting: {rel.n_supporting}")
        print(f"  Exemplar: {rel.exemplar}\n")
    
    stats = explicit.stats(relations)
    print("Statistics:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")



"""