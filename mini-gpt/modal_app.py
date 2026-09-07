"""
Run train.py on a Modal GPU.

One-time setup (see README's "GPU / Modal" section for details):
    pip install modal
    modal setup

Then, from this directory:
    modal run modal_app.py --steps 2000
    modal run modal_app.py --steps 2000 --d-model 512 --n-layers 8

Checkpoints land in a Modal Volume (persists across runs) and are also
downloaded to ./out/ on your machine when the run finishes.
"""

from __future__ import annotations

import modal

from config import DEFAULTS

app = modal.App("mini-gpt")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch").add_local_python_source(
    "config", "data", "model", "train"
)

volume = modal.Volume.from_name("mini-gpt-checkpoints", create_if_missing=True)
REMOTE_CHECKPOINT_DIR = "/checkpoints"


@app.function(image=image, gpu="T4", volumes={REMOTE_CHECKPOINT_DIR: volume}, timeout=3600)
def train_remote(args_dict: dict) -> None:
    import argparse

    from train import train

    train(argparse.Namespace(**args_dict))
    volume.commit()


@app.local_entrypoint()
def main(
    data_file: str = None,
    checkpoint: str = f"{REMOTE_CHECKPOINT_DIR}/ckpt.pt",
    d_model: int = DEFAULTS.d_model,
    n_heads: int = DEFAULTS.n_heads,
    n_layers: int = DEFAULTS.n_layers,
    max_seq_len: int = DEFAULTS.max_seq_len,
    ffn_dim: int = DEFAULTS.ffn_dim,
    dropout: float = DEFAULTS.dropout,
    batch_size: int = DEFAULTS.batch_size,
    block_size: int = DEFAULTS.block_size,
    lr: float = DEFAULTS.lr,
    steps: int = DEFAULTS.steps,
    log_every: int = DEFAULTS.log_every,
    seed: int = DEFAULTS.seed,
) -> None:
    args_dict = dict(
        data_file=data_file,
        checkpoint=checkpoint,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        ffn_dim=ffn_dim,
        dropout=dropout,
        batch_size=batch_size,
        block_size=block_size,
        lr=lr,
        steps=steps,
        log_every=log_every,
        seed=seed,
    )

    train_remote.remote(args_dict)

    local_path = f"out/{checkpoint.rsplit('/', 1)[-1]}"
    import os

    os.makedirs("out", exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in volume.read_file(checkpoint.removeprefix(f"{REMOTE_CHECKPOINT_DIR}/")):
            f.write(chunk)

    print(f"Downloaded checkpoint to {local_path}")
