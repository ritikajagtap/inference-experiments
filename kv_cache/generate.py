"""
Load a checkpoint and sample from it, with or without the KV cache.

    python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --max-new-tokens 100
    python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --no-cache
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
    """Full-recompute decoding: reprocess the whole sequence every step."""

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


@torch.no_grad()
def generate_cached(
    model: MiniGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """KV-cached decoding: run the prompt once (prefill), then feed one
    new token at a time, reusing the growing cache instead of reprocessing
    everything already generated."""

    logits, caches = model(input_ids, use_cache=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        logits, caches = model(next_token, caches=caches, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate(
    model: MiniGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
) -> torch.Tensor:
    if use_cache:
        return generate_cached(model, input_ids, max_new_tokens)
    return generate_naive(model, input_ids, max_new_tokens)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a MiniGPT checkpoint")

    p.add_argument("--checkpoint", type=str, default=DEFAULTS.checkpoint)
    add_generation_args(p)
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )

    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
    )

    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
