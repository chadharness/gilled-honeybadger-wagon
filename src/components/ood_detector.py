"""OODDetector — autoencoder-based out-of-distribution detection.

Combines autoencoder reconstruction error with error-weighted KNN on
labeled calibration embeddings. Neighbors are weighted by both embedding
similarity and reconstruction error proximity, so that neighbors with
similar autoencoder behavior have more influence.

Self-calibrates when an embedder is provided. Falls back to reconstruction
error threshold otherwise.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils.model_loader import load_model


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CALIBRATION_DATA_PATH = _PROJECT_ROOT / "data" / "ood_calibration_examples.csv"
_K = 7  # KNN neighborhood size
_ERROR_SCALE = 15.0  # Controls influence of error similarity in KNN weighting


@dataclass
class OODResult:
    is_ood: bool
    reconstruction_error: float
    threshold: float


class _Autoencoder(nn.Module):
    """Simple autoencoder: 384 → 64 → 384."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class OODDetector:
    """Autoencoder OOD detector with error-weighted KNN calibration.

    KNN neighbor weight = embedding_similarity * error_similarity, where
    error_similarity = 1 / (1 + |query_error - neighbor_error| * scale).
    This ensures neighbors with similar autoencoder reconstruction behavior
    have greater influence on the classification.
    """

    def __init__(self, embedder=None) -> None:
        model_data = load_model("ood_detector")
        if isinstance(model_data, dict) and "model_state_dict" in model_data:
            input_dim = model_data.get("input_dim", 384)
            hidden_dim = model_data.get("hidden_dim", 64)
            self._model = _Autoencoder(input_dim, hidden_dim)
            self._model.load_state_dict(model_data["model_state_dict"])
            self._model.eval()
        elif isinstance(model_data, dict) and "model" in model_data:
            self._model = model_data["model"]
        else:
            self._model = model_data

        self._cal_embeddings = []
        self._cal_labels = []
        self._cal_errors = []
        self._use_knn = False

        # Calibrate using labeled data if embedder available
        if embedder is not None and _CALIBRATION_DATA_PATH.exists():
            self._calibrate(embedder)
            self._use_knn = True
            self._threshold = 0.0  # Not used in KNN mode
        else:
            # Fall back to stored threshold on reconstruction error
            if isinstance(model_data, dict):
                mu = model_data.get("mu", 0.0)
                sigma = model_data.get("sigma", 0.0)
                lam = model_data.get("lambda", 4)
                self._threshold = mu + lam * sigma
            else:
                self._threshold = 0.0

    def _calibrate(self, embedder) -> None:
        """Embed calibration examples, compute errors, store for KNN."""
        with open(_CALIBRATION_DATA_PATH) as f:
            for row in csv.DictReader(f):
                emb = embedder.embed(row["text"])
                err = self._compute_error(emb)
                self._cal_embeddings.append(emb)
                self._cal_labels.append(row["label"])
                self._cal_errors.append(err)

    def _compute_error(self, embedding: np.ndarray) -> float:
        """Compute reconstruction error for an embedding."""
        with torch.no_grad():
            inp = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            reconstruction = self._model(inp)
            return torch.mean((inp - reconstruction) ** 2).item()

    def _knn_classify(self, embedding: np.ndarray, error: float) -> bool:
        """Classify using error-weighted KNN on calibration embeddings.

        Weight = embedding_similarity * error_proximity.
        Returns True if the query is OOD.
        """
        sims = np.array([float(np.dot(embedding, ce)) for ce in self._cal_embeddings])
        top_k_idx = np.argsort(sims)[-_K:]

        id_weight = 0.0
        ood_weight = 0.0
        for i in top_k_idx:
            sim_w = max(0.0, sims[i])
            err_diff = abs(error - self._cal_errors[i])
            err_w = 1.0 / (1.0 + err_diff * _ERROR_SCALE)
            w = sim_w * err_w
            if self._cal_labels[i] == "in_domain":
                id_weight += w
            else:
                ood_weight += w

        return ood_weight > id_weight

    def detect(self, embedding: np.ndarray) -> OODResult:
        """Check if an embedding is out-of-distribution.

        Args:
            embedding: 384-dim numpy array from SentenceEmbedder.

        Returns:
            OODResult with is_ood flag, reconstruction_error, and threshold.
        """
        error = self._compute_error(embedding)

        if self._use_knn:
            is_ood = self._knn_classify(embedding, error)
        else:
            # Fallback: threshold on reconstruction error
            is_ood = error > self._threshold

        return OODResult(
            is_ood=is_ood,
            reconstruction_error=error,
            threshold=self._threshold,
        )
