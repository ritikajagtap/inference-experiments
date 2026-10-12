"""
Compare generation throughput (tokens/sec) with and without the KV cache.

    python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 200

Add --sweep to also collect the data behind a proper comparison chart —
per-step decode latency (does it grow with sequence length, or stay
flat?) and a tok/s-vs-length sweep — dumped as JSON:

    python benchmark.py --checkpoint out/ckpt.pt --sweep --json out/bench.json

Without a checkpoint, an untrained model of the requested size is
benchmarked instead (throughput doesn't depend on the weights — only
architecture and sequence length do).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import flops as flops_model
from config import add_benchmark_args, add_flops_args, add_model_args, model_config_from_args
from data import CharTokenizer, DEFAULT_TEXT
from generate import generate_cached, generate_naive, load_checkpoint
from model import MiniGPT


def build_fresh_model(args: argparse.Namespace, device: torch.device) -> tuple[MiniGPT, CharTokenizer]:
    tokenizer = CharTokenizer(DEFAULT_TEXT)
    config = model_config_from_args(args, vocab_size=tokenizer.vocab_size)
    model = MiniGPT(config).to(device).eval()
    return model, tokenizer


def time_generation(
    generate_fn,
    model: MiniGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    generate_fn(model, input_ids, max_new_tokens=max_new_tokens)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


@torch.no_grad()
def naive_step_times(
    model: MiniGPT, input_ids: torch.Tensor, max_new_tokens: int, device: torch.device
) -> list[float]:
    """Wall-clock time for each individual decode step of naive (full
    resequence recompute) generation — expected to grow with step index,
    since each step reprocesses a longer sequence."""

    times = []
    for _ in range(max_new_tokens):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits, _ = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


@torch.no_grad()
def cached_step_times(
    model: MiniGPT, input_ids: torch.Tensor, max_new_tokens: int, device: torch.device
) -> list[float]:
    """Wall-clock time for each individual decode step with the KV cache.
    times[0] is prefill (processes the whole prompt) + the first token;
    times[1:] are single-token decode steps, expected to stay ~flat
    regardless of how long the sequence has grown."""

    times = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    logits, caches = model(input_ids, use_cache=True)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

    for _ in range(max_new_tokens - 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits, caches = model(next_token, caches=caches, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


def length_sweep(
    model: MiniGPT,
    input_ids: torch.Tensor,
    lengths: list[int],
    device: torch.device,
) -> dict:
    cached_tok_s, naive_tok_s = [], []
    for n in lengths:
        cached_time = time_generation(generate_cached, model, input_ids, n, device)
        naive_time = time_generation(generate_naive, model, input_ids, n, device)
        cached_tok_s.append(n / cached_time)
        naive_tok_s.append(n / naive_time)
    return {"lengths": lengths, "cached_tok_s": cached_tok_s, "naive_tok_s": naive_tok_s}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark cached vs uncached generation")

    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--sweep", action="store_true", help="also collect per-step and length-sweep data")
    p.add_argument("--flops", action="store_true",
                   help="also print analytical FLOPs: cached vs naive")
    p.add_argument("--json", type=str, default=None, help="write --sweep/--flops results to this path")

    add_benchmark_args(p)
    add_flops_args(p)
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
    time_generation(generate_cached, model, input_ids, args.warmup_tokens, device)
    time_generation(generate_naive, model, input_ids, args.warmup_tokens, device)

    cached_time = time_generation(generate_cached, model, input_ids, args.max_new_tokens, device)
    uncached_time = time_generation(generate_naive, model, input_ids, args.max_new_tokens, device)

    cached_tps = args.max_new_tokens / cached_time
    uncached_tps = args.max_new_tokens / uncached_time

    print()
    print(f"{'mode':<10} {'time (s)':>10} {'tok/s':>10}")
    print(f"{'cached':<10} {cached_time:>10.3f} {cached_tps:>10.1f}")
    print(f"{'uncached':<10} {uncached_time:>10.3f} {uncached_tps:>10.1f}")
    print()
    print(f"Speedup: {cached_tps / uncached_tps:.2f}x at {args.max_new_tokens} generated tokens")

    results = {"device": str(device), "prompt": args.prompt}

    if args.sweep:
        print()
        print("Collecting per-step latency...")
        step_count = args.max_new_tokens
        naive_times = naive_step_times(model, input_ids, step_count, device)
        cached_times = cached_step_times(model, input_ids, step_count, device)

        print("Running length sweep...")
        lengths = sorted({max(1, step_count // 8), step_count // 4, step_count // 2, step_count})
        sweep = length_sweep(model, input_ids, lengths, device)

        results.update(
            step_times_naive=naive_times,
            step_times_cached=cached_times,
            length_sweep=sweep,
        )

    if args.flops:
        prompt_len = input_ids.shape[1]
        config = model.config
        rows = flops_model.sweep(config, prompt_len, args.seq_lens)

        G = 1e9
        print()
        print("Analytical FLOPs (cached vs naive), generating N new tokens:")
        print(f"{'new tokens':>10} {'naive GFLOPs':>14} {'cached GFLOPs':>14} {'naive/cached':>14}")
        for r in rows:
            print(f"{r['n_new']:>10} {r['naive_flops']/G:>14.2f} "
                  f"{r['cached_flops']/G:>14.2f} {r['ratio']:>13.1f}x")
        print("Naive is quadratic in length, cached ~linear - the ratio widens with N.")

        results["flops"] = {
            "prompt_len": prompt_len,
            "matmul_params": flops_model.param_count(config),
            "sweep": rows,
        }

    if args.json and (args.sweep or args.flops):
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f)
        print(f"\nWrote results to {args.json}")


if __name__ == "__main__":
    main()
