import numpy as np
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
import spacy

from config.llm import GeminiConfig
from llm.prompts.gemini import RELATION_PROMPT
from nlp.syntactic import extract_exemplar, mask_entities, find_entity_span


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
    """Explicit relation discovery via dependency extraction + AAP + Gemini."""

    def __init__(
        self,
        config: GeminiConfig,
        encoder_model: str = 'all-mpnet-base-v2',
        damping: float = 0.9,
        max_iter: int = 200,
        conv_iter: int = 10
    ):
        """Initialize with MPNet encoder and Gemini labeler.
        
        Args:
            config: Gemini API configuration
            encoder_model: SentenceTransformer model for AAP clustering
            damping: AP damping factor (0.5–1.0)
            max_iter: Maximum AP iterations
            conv_iter: Iterations before checking convergence
        """
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.encoder_model = encoder_model
        self.damping = damping
        self.max_iter = max_iter
        self.conv_iter = conv_iter
        self._encoder = None
        self.nlp = spacy.load("en_core_web_sm")

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load sentence encoder."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.encoder_model)
        return self._encoder

    def discover(self, candidates, sentences, entities=None):
        explicit_rels = []
        full_text = " ".join(sentences)
        doc = self.nlp(full_text)
        
        exemplar_data = []
        
        for cand in candidates:
            e1 = cand.e1 if hasattr(cand, 'e1') else cand['e1']
            e2 = cand.e2 if hasattr(cand, 'e2') else cand['e2']
            
            for sent in doc.sents:
                sent_text = sent.text
                if e1.lower() not in sent_text.lower():
                    continue
                if e2.lower() not in sent_text.lower():
                    continue
                
                exemplar = extract_exemplar(doc, e1, e2)
                if not exemplar:
                    exemplar = sent_text
                
                masked = mask_entities(exemplar, e1, e2)
                if not masked:
                    continue
                
                support = sum(1 for s in sentences 
                            if e1.lower() in s.lower() and e2.lower() in s.lower())
                
                exemplar_data.append((e1, e2, exemplar, masked, support))
                break
        
        if not exemplar_data:
            return []
        
        exemplars_only = [x[3] for x in exemplar_data]
        exemplar_embeddings = self.encoder.encode(exemplars_only, show_progress_bar=False, convert_to_numpy=True)
        exemplar_clusters = self._aap_cluster(exemplar_embeddings)
        
        for i, (e1, e2, orig_exemplar, masked, support) in enumerate(exemplar_data):
            rel_type = self._label_relation(masked)
            confidence = min(support / 10.0, 1.0)
            
            explicit_rels.append(ExplicitRelation(
                e1=e1,
                e2=e2,
                rel_type=rel_type,
                confidence=confidence,
                exemplar=masked,
                n_supporting=support
            ))
        
        return sorted(explicit_rels, key=lambda x: x.confidence, reverse=True)



    def _aap_cluster(self, embeddings: np.ndarray) -> dict:
        """Run Affinity Propagation, return cluster assignments.
        
        Clusters exemplar embeddings to identify representative archetypes.
        Exemplars within cluster are semantically similar expression patterns.
        
        Returns:
            dict mapping exemplar index → cluster exemplar index
        """
        m = len(embeddings)
        if m == 1:
            return {0: 0}
        
        # similarity matrix (cosine distance)
        S = cosine_similarity(embeddings)
        r = np.zeros((m, m), dtype=np.float32)  # responsibility
        a = np.zeros((m, m), dtype=np.float32)  # availability
        
        for iteration in range(self.max_iter):
            r_old = r.copy()
            a_old = a.copy()
            
            # responsibility: how well k represents i
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
            
            # availability: how suitable i is as exemplar
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
                exemplars = np.array([np.argmax(a[i, :] + r[i, :]) for i in range(m)])
                if iteration > self.conv_iter:
                    exemplars_prev = np.array([np.argmax(a_old[i, :] + r_old[i, :]) for i in range(m)])
                    if np.array_equal(exemplars, exemplars_prev):
                        break
        
        # final cluster assignments
        exemplars = np.array([np.argmax(a[i, :] + r[i, :]) for i in range(m)])
        return {i: int(exemplars[i]) for i in range(m)}

    def _label_relation(self, exemplar: str) -> str:
        """Classify relation type with Gemini, fallback to keywords."""
        try:
            config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=50
            )
            
            prompt = RELATION_PROMPT.format(exemplar=exemplar)
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config=config
            )
            
            if response and hasattr(response, 'text') and response.text:
                rel_type = response.text.strip().lower()
                valid = {'uses', 'improves', 'evaluates', 'enables', 'proposes', 'related'}
                return rel_type if rel_type in valid else 'related'
        
        except Exception:
            pass
        
        return self._label_fallback(exemplar)

    def _label_fallback(self, exemplar: str) -> str:
        """Fallback keyword-based classification."""
        exemplar_lower = exemplar.lower()
        
        patterns = {
            'uses': ['use', 'employ', 'apply', 'utilize', 'integrate'],
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
        """Compute summary statistics over relations."""
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
