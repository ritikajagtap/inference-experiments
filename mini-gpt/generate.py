"""
Load a checkpoint and sample from it.

    python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --max-new-tokens 100
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from config import DEFAULTS, add_generation_args
from data import CharTokenizer
from model import GPTConfig, MiniGPT


def load_checkpoint(path: str, device: torch.device) -> tuple[MiniGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location=device)

    config = GPTConfig(**ckpt["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.from_state_dict(ckpt["tokenizer"])

    return model, tokenizer


@torch.no_grad()
def generate_naive(
    model: MiniGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # Run the entire sequence through the model
        logits = model(input_ids)

        # Only care about the last position
        logits = logits[:, -1, :]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Pick the most likely token
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # Add it to our sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a MiniGPT checkpoint")

    p.add_argument("--checkpoint", type=str, default=DEFAULTS.checkpoint)
    add_generation_args(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )

    generated_ids = generate_naive(model, input_ids, max_new_tokens=args.max_new_tokens)

    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
