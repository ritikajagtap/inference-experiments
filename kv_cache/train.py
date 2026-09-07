"""
Training loop for MiniGPT. Identical to ../train.py — training never
touches the KV cache (that's a generation-time optimization), only the
extra `caches` return value from `model()` needs unpacking.

Run locally with:
    python train.py --steps 2000 --checkpoint out/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import DEFAULTS, add_model_args, add_train_args, model_config_from_args
from data import CharTokenizer, get_batch, load_text
from model import GPTConfig, MiniGPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniGPT")

    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=DEFAULTS.checkpoint)

    add_model_args(p)
    add_train_args(p)

    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    text = load_text(args.data_file)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = model_config_from_args(args, vocab_size=tokenizer.vocab_size)

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    loss_history = []
    start_time = time.time()

    for step in range(args.steps):
        x, y = get_batch(data, args.batch_size, args.block_size, device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

    print(f"Final loss: {loss_history[-1]:.4f}")

    save_checkpoint(args.checkpoint, model, config, tokenizer)
    print(f"Saved checkpoint to {args.checkpoint}")

    history_path = save_loss_history(args.checkpoint, loss_history)
    print(f"Saved loss history to {history_path}")


def save_checkpoint(path: str, model: MiniGPT, config: GPTConfig, tokenizer: CharTokenizer) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "tokenizer": tokenizer.state_dict(),
        },
        path,
    )


def save_loss_history(checkpoint_path: str, loss_history: list[float]) -> str:
    """Write loss_history as JSON next to the checkpoint (e.g. out/ckpt.pt
    -> out/ckpt.loss.json), for later plotting."""

    path = Path(checkpoint_path).with_suffix("").with_suffix(".loss.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(loss_history, f)
    return str(path)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
