"""Model loading abstraction — reads models.yaml and loads local models.

For local-huggingface: loads via sentence-transformers or setfit.
For local-weights: loads via torch.load.
For portkey-gateway: returns config dict (actual client in llm_client.py).
"""

from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "models.yaml"

_loaded_models: dict[str, Any] = {}


def _load_config() -> dict[str, Any]:
    with open(_MODELS_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the configuration dict for a named model entry."""
    config = _load_config()
    entries = config.get("entries", {})
    if model_name not in entries:
        raise KeyError(f"Model '{model_name}' not found in models.yaml")
    return entries[model_name]


def load_model(model_name: str, force_reload: bool = False) -> Any:
    """Load a model by name. Caches loaded models.

    Returns:
        For local-huggingface (sentence-transformers): SentenceTransformer instance
        For local-huggingface (setfit): SetFitModel instance
        For local-weights (torch): loaded state dict / model
        For portkey-gateway: config dict (use llm_client for actual calls)
    """
    if model_name in _loaded_models and not force_reload:
        return _loaded_models[model_name]

    entry = get_model_config(model_name)
    connection = entry["connection"]
    artifact_path = str(_PROJECT_ROOT / entry["artifact_path"]) if "artifact_path" in entry else None

    if connection == "local-huggingface":
        runtime = entry.get("runtime", "")
        if runtime == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(artifact_path)
        elif runtime == "setfit":
            from setfit import SetFitModel
            model = SetFitModel.from_pretrained(artifact_path)
        else:
            raise ValueError(f"Unknown runtime '{runtime}' for local-huggingface model '{model_name}'")
    elif connection == "local-weights":
        import torch
        # weights_only=False required: autoencoder saved as dict with model + threshold
        model = torch.load(artifact_path, map_location="cpu", weights_only=False)
    elif connection == "portkey-gateway":
        model = {
            "model_id": entry["model_id"],
            "provider": entry.get("provider", ""),
            "portkey_model": entry.get("portkey_model", ""),
            "connection": "portkey-gateway",
        }
    else:
        raise ValueError(f"Unknown connection type '{connection}' for model '{model_name}'")

    _loaded_models[model_name] = model
    return model
