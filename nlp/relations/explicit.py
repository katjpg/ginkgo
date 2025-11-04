import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import re
from google import genai
from google.genai import types
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
    
    def __init__(
        self,
        config: GeminiConfig,
        encoder_model: str = 'all-mpnet-base-v2',
        damping: float = 0.9,
        max_iter: int = 200,
        conv_iter: int = 10
    ):
        """Initialize with MPNet encoder and Gemini labeler."""
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.encoder_model = encoder_model
        self.damping = damping
        self.max_iter = max_iter
        self.conv_iter = conv_iter
        self._encoder = None
    
    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load sentence encoder."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.encoder_model)
        return self._encoder
    
    def discover(
        self,
        candidates: list,
        sentences: list[str],
        entities: list[dict] | None = None
    ) -> list[ExplicitRelation]:
        """Discover explicit relations from candidate pairs."""
        explicit_rels = []
        
        for cand in candidates:
            e1 = cand.e1 if hasattr(cand, 'e1') else cand['e1']
            e2 = cand.e2 if hasattr(cand, 'e2') else cand['e2']
            
            masked_sents = self._extract_masked_sentences(e1, e2, sentences)
            
            if not masked_sents:
                continue
            
            if len(masked_sents) == 1:
                exemplar = masked_sents[0]
            else:
                embeddings = self.encoder.encode(
                    masked_sents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                exemplar_idx = self._aap_cluster(embeddings)
                exemplar = masked_sents[exemplar_idx]
            
            rel_type = self._label_with_gemini(exemplar)
            confidence = min(len(masked_sents) / 10.0, 1.0)
            
            explicit_rels.append(ExplicitRelation(
                e1=e1,
                e2=e2,
                rel_type=rel_type,
                confidence=confidence,
                exemplar=exemplar,
                n_supporting=len(masked_sents)
            ))
        
        return sorted(explicit_rels, key=lambda x: x.confidence, reverse=True)
    
    def _extract_masked_sentences(
        self,
        e1: str,
        e2: str,
        sentences: list[str]
    ) -> list[str]:
        """Extract sentences with both entities, replace with [ENT1]/[ENT2]."""
        masked = []
        e1_lower = e1.lower()
        e2_lower = e2.lower()
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            if e1_lower not in sent_lower or e2_lower not in sent_lower:
                continue
            
            # case-insensitive replacement
            masked_sent = re.sub(re.escape(e1), '[ENT1]', sent, flags=re.IGNORECASE, count=1)
            masked_sent = re.sub(re.escape(e2), '[ENT2]', masked_sent, flags=re.IGNORECASE, count=1)
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
            
            # update responsibilities
            for i in range(m):
                for k in range(m):
                    mask = np.ones(m, dtype=bool)
                    mask[k] = False
                    
                    if mask.any():
                        max_val = np.max(a[i, mask] + S[i, mask])
                        r[i, k] = S[i, k] - max_val
                    else:
                        r[i, k] = S[i, k]
            
            r = self.damping * r + (1 - self.damping) * r_old
            
            # update availabilities
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
                        a[i, k] = min(0, r[k, k] + sum_val)
            
            a = self.damping * a + (1 - self.damping) * a_old
            
            # check convergence
            if iteration >= self.conv_iter:
                exemplars_curr = np.array([np.argmax(a[i, :] + r[i, :]) for i in range(m)])
                if iteration > self.conv_iter:
                    exemplars_prev = np.array([np.argmax(a_old[i, :] + r_old[i, :]) for i in range(m)])
                    if np.array_equal(exemplars_curr, exemplars_prev):
                        break
        
        exemplar_idx = np.argmax(np.diag(a + r))
        return int(exemplar_idx)
    
    def _label_with_gemini(self, exemplar: str) -> str:
        """Use Gemini to classify relation type from exemplar."""
        prompt = RELATION_PROMPT.format(exemplar=exemplar)
        
        try:
            config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=50,  # short output needed
            )
            
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config=config
            )
            
            # handle potential None response
            if response and hasattr(response, 'text') and response.text:
                rel_type = response.text.strip().lower()
            else:
                return self._label_with_keyword(exemplar)
            
            valid_types = {'uses', 'improves', 'evaluates', 'enables', 'proposes', 'related'}
            return rel_type if rel_type in valid_types else 'related'
        
        except Exception:
            # use keyword matching if API fails
            return self._label_with_keyword(exemplar)
    
    def _label_with_keyword(self, exemplar: str) -> str:
        """Keyword-based classification."""
        exemplar_lower = exemplar.lower()
        
        patterns = {
            'uses': ['use', 'employ', 'apply', 'utilize', 'leverage'],
            'improves': ['improve', 'enhance', 'outperform', 'boost'],
            'evaluates': ['evaluate', 'test', 'benchmark', 'assess'],
            'enables': ['enable', 'allow', 'facilitate', 'support'],
            'proposes': ['propose', 'introduce', 'present']
        }
        
        for rel_type, keywords in patterns.items():
            for kw in keywords:
                if kw in exemplar_lower:
                    return rel_type
        
        return 'related'
    
    def stats(self, relations: list[ExplicitRelation]) -> dict:
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
        type_dist = {}
        
        for r in relations:
            type_dist[r.rel_type] = type_dist.get(r.rel_type, 0) + 1
        
        return {
            'n_explicit': len(relations),
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'mean_support': float(np.mean(supports)),
            'rel_type_dist': type_dist
        }