"""SentenceEmbedder — loads from model_registry, produces embeddings."""

import numpy as np

from src.utils.model_loader import load_model


class SentenceEmbedder:
    """Wraps a SentenceTransformer model loaded from models.yaml."""

    def __init__(self) -> None:
        self._model = load_model("embedder")

    def embed(self, text: str) -> np.ndarray:
        """Produce a 384-dim embedding for a single text string."""
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Produce embeddings for a batch of texts."""
        return self._model.encode(texts, normalize_embeddings=True)
