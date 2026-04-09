"""AgentRouter — SetFit-based binary classifier for routing queries.

Routes to: data_presenter or insight_generator.
Uses the shared embedding from SentenceEmbedder when available, falling
back to SetFit's internal encoding when called with text only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.model_loader import load_model


@dataclass
class RouteDecision:
    agent: str  # "data_presenter" or "insight_generator"
    confidence: float


class AgentRouter:
    """SetFit binary classifier for agent routing."""

    def __init__(self) -> None:
        self._model = load_model("agent_router")
        self._head = self._model.model_head
        self._id2label = getattr(self._model, "id2label", {0: "data_presenter", 1: "insight_generator"})

    def route(self, text: str, embedding: np.ndarray | None = None) -> RouteDecision:
        """Route a query to the appropriate worker agent.

        Args:
            text: The user query or augmented query.
            embedding: Pre-computed 384-dim embedding from shared SentenceEmbedder.
                       When provided, classification head is called directly
                       (no re-encoding). Falls back to full SetFit predict otherwise.

        Returns:
            RouteDecision with agent name and confidence.
        """
        if embedding is not None:
            return self._route_from_embedding(embedding)
        return self._route_from_text(text)

    def _route_from_embedding(self, embedding: np.ndarray) -> RouteDecision:
        """Classify using the model head directly on a pre-computed embedding."""
        x = embedding.reshape(1, -1)
        label_id = int(self._head.predict(x)[0])
        agent = self._id2label.get(label_id, "data_presenter")

        confidence = 1.0
        try:
            proba = self._head.predict_proba(x)
            confidence = float(np.max(proba[0]))
        except (AttributeError, RuntimeError):
            pass  # Not all model heads support predict_proba; default to 1.0 for now

        return RouteDecision(agent=agent, confidence=confidence)

    def _route_from_text(self, text: str) -> RouteDecision:
        """Classify using full SetFit pipeline (encodes internally)."""
        prediction = self._model.predict([text])
        label = prediction[0] if hasattr(prediction, '__len__') else prediction

        confidence = 1.0
        try:
            proba = self._model.predict_proba([text])
            if hasattr(proba, 'shape') and len(proba.shape) > 1:
                proba_row = proba[0]
                if hasattr(proba_row, 'numpy'):
                    proba_row = proba_row.numpy()
                confidence = float(np.max(proba_row))
        except (AttributeError, RuntimeError):
            pass  # Not all SetFit versions expose predict_proba; default to 1.0 for now

        agent = str(label)
        return RouteDecision(agent=agent, confidence=confidence)
