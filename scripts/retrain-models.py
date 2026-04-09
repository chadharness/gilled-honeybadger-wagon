#!/usr/bin/env python3
"""
Retrain OOD autoencoder and SetFit router in the current venv.

Ensures model weights are compatible with the exact torch and
sentence-transformers versions installed. Reads training data from
data/ and saves models to models/.

Usage:
    PYTHONPATH=. python3 scripts/retrain-models.py

This should be run in the engagement repo's venv BEFORE the build,
typically during /setup-and-validate (stage 2).
"""

import os
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
EMBEDDER_DIR = MODEL_DIR / "embedder"

EMBEDDER_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
HIDDEN_DIM = 64
LAMBDA = 4
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 32


def retrain_ood():
    """Retrain OOD autoencoder with current sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("Retraining OOD Autoencoder")
    print("=" * 60)

    # Load embedder (and re-save to ensure compatibility)
    print(f"Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    EMBEDDER_DIR.mkdir(parents=True, exist_ok=True)
    embedder.save(str(EMBEDDER_DIR))
    print(f"Embedder saved to {EMBEDDER_DIR}")

    # Load training data
    ood_train_df = pd.read_csv(DATA_DIR / "ood_training_questions.csv")
    in_domain_texts = ood_train_df[ood_train_df["label"] == "in_domain"]["text"].tolist()
    print(f"Training data: {len(in_domain_texts)} in-domain texts")

    # Embed
    print("Embedding training data...")
    in_domain_embeddings = embedder.encode(in_domain_texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize to [0,1] for sigmoid compatibility
    id_tensor = torch.tensor(in_domain_embeddings, dtype=torch.float32)
    embed_min = id_tensor.min(dim=0).values
    embed_max = id_tensor.max(dim=0).values
    embed_range = embed_max - embed_min
    embed_range[embed_range == 0] = 1.0
    id_tensor_norm = (id_tensor - embed_min) / embed_range

    # Autoencoder
    class OODAutoencoder(nn.Module):
        def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = OODAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    dataset = TensorDataset(id_tensor_norm)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training: {EMBEDDING_DIM}->{HIDDEN_DIM}->{EMBEDDING_DIM}, {EPOCHS} epochs")
    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(id_tensor_norm)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {epoch_loss:.6f}")

    print(f"Training complete in {time.time() - t0:.1f}s")

    # Compute threshold
    model.eval()
    with torch.no_grad():
        recon_id = model(id_tensor_norm)
        recon_errors = ((recon_id - id_tensor_norm) ** 2).mean(dim=1).numpy()

    mu = recon_errors.mean()
    sigma = recon_errors.std()
    threshold = mu + LAMBDA * sigma

    print(f"Threshold: {threshold:.6f} (mu={mu:.6f} + {LAMBDA}*sigma={sigma:.6f})")

    # Save
    save_path = MODEL_DIR / "ood_autoencoder.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "lambda": LAMBDA,
        "mu": mu,
        "sigma": sigma,
        "embed_min": embed_min.numpy(),
        "embed_max": embed_max.numpy(),
        "embed_range": embed_range.numpy(),
        "input_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_training_samples": len(in_domain_texts),
    }, save_path)
    print(f"Saved to {save_path}")

    # Quick validation
    cal_path = DATA_DIR / "ood_calibration_examples.csv"
    if cal_path.exists():
        cal_df = pd.read_csv(cal_path)
        cal_in = cal_df[cal_df["label"] == "in_domain"]["text"].tolist()
        cal_out = cal_df[cal_df["label"] == "out_of_domain"]["text"].tolist()

        cal_in_emb = torch.tensor(embedder.encode(cal_in, convert_to_numpy=True), dtype=torch.float32)
        cal_out_emb = torch.tensor(embedder.encode(cal_out, convert_to_numpy=True), dtype=torch.float32)

        cal_in_norm = (cal_in_emb - embed_min) / embed_range
        cal_out_norm = (cal_out_emb - embed_min) / embed_range

        with torch.no_grad():
            in_errors = ((model(cal_in_norm) - cal_in_norm) ** 2).mean(dim=1).numpy()
            out_errors = ((model(cal_out_norm) - cal_out_norm) ** 2).mean(dim=1).numpy()

        in_correct = sum(1 for e in in_errors if e <= threshold)
        out_correct = sum(1 for e in out_errors if e > threshold)
        print(f"\nValidation: {in_correct}/{len(cal_in)} in-domain correct, {out_correct}/{len(cal_out)} out-domain correct")

    return embedder


def retrain_router(embedder):
    """Retrain SetFit router with current sentence-transformers."""
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.model_selection import train_test_split

    print()
    print("=" * 60)
    print("Retraining SetFit Router")
    print("=" * 60)

    router_df = pd.read_csv(DATA_DIR / "router_training_examples.csv")
    print(f"Training data: {len(router_df)} rows")

    label_map = {"data_presenter": 0, "insight_generator": 1}
    router_df["label_id"] = router_df["label"].map(label_map)

    train_df, eval_df = train_test_split(
        router_df, test_size=0.2, random_state=SEED, stratify=router_df["label_id"]
    )

    train_dataset = Dataset.from_dict({
        "text": train_df["text"].tolist(),
        "label": train_df["label_id"].tolist()
    })
    eval_dataset = Dataset.from_dict({
        "text": eval_df["text"].tolist(),
        "label": eval_df["label_id"].tolist()
    })

    setfit_model = SetFitModel.from_pretrained(
        EMBEDDER_NAME,
        labels=["data_presenter", "insight_generator"]
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        seed=SEED,
    )

    trainer = Trainer(
        model=setfit_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training SetFit router...")
    t0 = time.time()
    trainer.train()
    print(f"Training complete in {time.time() - t0:.1f}s")

    metrics = trainer.evaluate()
    print(f"Eval accuracy: {metrics['accuracy']:.3f}")

    router_save_path = MODEL_DIR / "router_setfit"
    setfit_model.save_pretrained(str(router_save_path))
    print(f"Saved to {router_save_path}")


if __name__ == "__main__":
    print(f"Python: {os.sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"Working dir: {ROOT}")
    print()

    embedder = retrain_ood()
    retrain_router(embedder)

    print()
    print("=" * 60)
    print("All models retrained with current venv.")
    print("Models are now compatible with the build environment.")
    print("=" * 60)
