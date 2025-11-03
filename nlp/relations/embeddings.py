import numpy as np
from sentence_transformers import SentenceTransformer


class EntityEmbeddings:
    """Compute entity embeddings for semantic similarity."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
        strict: bool = True
    ):
        """Initialize embedding model."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.strict = strict
        self._model = None
        self._device = device
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model
    
    def compute(self, entity_texts: list[str]) -> np.ndarray:
        """Compute embeddings for entity list."""
        if not entity_texts:
            return np.array([], dtype=np.float32).reshape(0, self.dim())
        
        # validate and clean inputs
        cleaned = []
        indices = []
        for i, text in enumerate(entity_texts):
            if not text or not text.strip():
                if self.strict:
                    raise ValueError(f"Empty entity text at index {i}")
                continue
            cleaned.append(text.strip())
            indices.append(i)
        
        if not cleaned:
            return np.array([], dtype=np.float32).reshape(0, self.dim())
        
        embeddings = self.model.encode(
            cleaned,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # align output with input indices if filtering occurred
        if not self.strict and len(indices) < len(entity_texts):
            aligned = np.zeros((len(entity_texts), self.dim()), dtype=np.float32)
            for j, idx in enumerate(indices):
                aligned[idx] = embeddings[j]
            return aligned
        
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Compute embedding for single entity."""
        if not text or not text.strip():
            if self.strict:
                raise ValueError("Empty entity text")
            return np.zeros(self.dim(), dtype=np.float32)
        
        embedding = self.model.encode(
            text.strip(),
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding.astype(np.float32)
    
    def dim(self) -> int:
        """Return embedding dimension."""
        dim = self.model.get_sentence_embedding_dimension()
        assert dim is not None, f"Model {self.model_name} has no dimension"
        return dim