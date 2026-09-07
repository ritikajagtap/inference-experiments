"""
Measure generation throughput (tokens/sec) for MiniGPT's naive
(full-recompute) autoregressive decoding.

    python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 200

Note: this model has no KV cache (see README) — it always reruns the
whole sequence through every layer to produce each new token. This
script reports that baseline throughput at a couple of prompt lengths so
a future KV-cache implementation has a number to beat.

Without a checkpoint, an untrained model of the requested size is
benchmarked instead (throughput doesn't depend on the weights).
"""

from __future__ import annotations

import argparse
import time

import torch

from config import add_benchmark_args, add_model_args, model_config_from_args
from data import CharTokenizer, DEFAULT_TEXT
from generate import generate_naive, load_checkpoint
from model import MiniGPT


def build_fresh_model(args: argparse.Namespace, device: torch.device) -> tuple[MiniGPT, CharTokenizer]:
    tokenizer = CharTokenizer(DEFAULT_TEXT)
    config = model_config_from_args(args, vocab_size=tokenizer.vocab_size)
    model = MiniGPT(config).to(device).eval()
    return model, tokenizer


def time_generation(
    model: MiniGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    generate_naive(model, input_ids, max_new_tokens=max_new_tokens)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark naive generation throughput")

    p.add_argument("--checkpoint", type=str, default=None)

    add_benchmark_args(p)
    add_model_args(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.checkpoint is not None:
        model, tokenizer = load_checkpoint(args.checkpoint, device)
    else:
        model, tokenizer = build_fresh_model(args, device)

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )

    # Warm up (CUDA kernel compilation / caching allocator) before timing.
    time_generation(model, input_ids, args.warmup_tokens, device)

    elapsed = time_generation(model, input_ids, args.max_new_tokens, device)
    tokens_per_sec = args.max_new_tokens / elapsed

    print()
    print(f"{'mode':<10} {'time (s)':>10} {'tok/s':>10}")
    print(f"{'naive':<10} {elapsed:>10.3f} {tokens_per_sec:>10.1f}")


if __name__ == "__main__":
    main()
