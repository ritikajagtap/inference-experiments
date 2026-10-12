"""
Analytical FLOPs: cached vs naive generation.

Counts the arithmetic each generation path performs, straight from
`GPTConfig` — no tensors allocated, no checkpoint required, so it works at
any sequence length (including well past `max_seq_len`). A multiply-add is
counted as 2 FLOPs, the usual convention.

The point is the *scaling*:

- Naive generation re-feeds the whole growing sequence through the model at
  every decode step, so generating N tokens costs `Σ forward(P + i)` —
  quadratic in sequence length.
- Cached generation runs one prefill over the prompt, then cheap
  single-token decode steps (each attending to a KV cache that grows by
  one), so it costs `forward(P) + Σ decode_step(P + j)` — linear.

The step accounting here deliberately mirrors the loops in benchmark.py
(`naive_step_times` / `cached_step_times`), so these FLOPs describe the same
work the wall-clock numbers measure.

Usage:
    python flops.py
    python flops.py --seq-lens 128,512,2048 --json out/flops.json
    python flops.py --checkpoint out/ckpt.pt        # read config from a ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import add_flops_args, add_model_args, model_config_from_args
from data import DEFAULT_TEXT, CharTokenizer


def forward_flops_breakdown(config, T: int) -> dict[str, int]:
    """FLOPs for one forward pass over T tokens, by component.

    C = d_model, F = ffn_dim, V = vocab_size, L = n_layers. The token
    embedding is a lookup, not a matmul, so it contributes no FLOPs.
    """

    C = config.d_model
    F = config.ffn_dim
    V = config.vocab_size
    L = config.n_layers

    qkv_proj = L * 6 * T * C * C          # (T,C) x (C,3C)
    attention = L * 4 * T * T * C          # QKᵀ  +  (softmax·V), both 2·T²·C
    out_proj = L * 2 * T * C * C           # (T,C) x (C,C)
    ffn = L * 6 * T * C * F                # SwiGLU: gate + up + down matmuls
    lm_head = 2 * T * C * V                # (T,C) x (C,V)

    return {
        "qkv_proj": qkv_proj,
        "attention": attention,
        "out_proj": out_proj,
        "ffn": ffn,
        "lm_head": lm_head,
    }


def forward_flops(config, T: int) -> int:
    """Total FLOPs for one forward pass over T tokens."""

    return sum(forward_flops_breakdown(config, T).values())


def decode_step_flops(config, t_ctx: int) -> int:
    """FLOPs for one cached decode step: a single new token through the
    projections / FFN / lm_head (T=1), plus attention against a context of
    length `t_ctx` (the cache, including the new token)."""

    C = config.d_model
    F = config.ffn_dim
    V = config.vocab_size
    L = config.n_layers

    per_token = L * (6 * C * C + 2 * C * C + 6 * C * F) + 2 * C * V
    attention = L * 4 * t_ctx * C          # 1 query · t_ctx keys, scores + V
    return per_token + attention


def naive_total_flops(config, prompt_len: int, n_new: int) -> int:
    """Naive generation: step i reprocesses the full sequence of length
    (prompt_len + i). Matches benchmark.naive_step_times."""

    return sum(forward_flops(config, prompt_len + i) for i in range(n_new))


def cached_total_flops(config, prompt_len: int, n_new: int) -> int:
    """Cached generation: one prefill over the prompt, then (n_new - 1)
    single-token decode steps with the cache growing from prompt_len + 1.
    Matches benchmark.cached_step_times."""

    total = forward_flops(config, prompt_len)   # prefill + first token
    total += sum(decode_step_flops(config, prompt_len + j) for j in range(1, n_new))
    return total


def param_count(config, matmul_only: bool = True) -> int:
    """Parameter count. With matmul_only=True, excludes the token-embedding
    table (a lookup, not a matmul) and norm/bias terms — i.e. the N in the
    '~2N FLOPs per token' rule of thumb."""

    C = config.d_model
    F = config.ffn_dim
    V = config.vocab_size
    L = config.n_layers

    per_layer = 3 * C * C + C * C + 3 * C * F   # qkv + out_proj + SwiGLU
    total = L * per_layer + C * V               # + lm_head
    if not matmul_only:
        total += V * C                          # token embedding
    return total


def compare(config, prompt_len: int, n_new: int) -> dict:
    """Naive vs cached totals (and naive's component breakdown) for
    generating n_new tokens from a prompt of prompt_len tokens."""

    naive = naive_total_flops(config, prompt_len, n_new)
    cached = cached_total_flops(config, prompt_len, n_new)
    return {
        "prompt_len": prompt_len,
        "n_new": n_new,
        "naive_flops": naive,
        "cached_flops": cached,
        "ratio": naive / cached if cached else float("inf"),
    }


def sweep(config, prompt_len: int, lengths: list[int]) -> list[dict]:
    return [compare(config, prompt_len, n) for n in lengths]


def _sanity_check(config, prompt_len: int) -> None:
    """Cheap invariants that guard the formulas."""

    # forward over 1 token, minus the activation-dependent attention term,
    # is the classic 2N (N = matmul params).
    fwd1 = forward_flops_breakdown(config, 1)
    non_attn = sum(v for k, v in fwd1.items() if k != "attention")
    assert non_attn == 2 * param_count(config, matmul_only=True), (
        non_attn, 2 * param_count(config))

    # Generating exactly one token is a single prompt-length forward pass,
    # identical for both paths.
    assert naive_total_flops(config, prompt_len, 1) == forward_flops(config, prompt_len)
    assert cached_total_flops(config, prompt_len, 1) == forward_flops(config, prompt_len)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytical FLOPs: cached vs naive generation")

    p.add_argument("--checkpoint", type=str, default=None,
                   help="read model config from this checkpoint instead of --d-model etc.")
    p.add_argument("--prompt", type=str, default="the cat",
                   help="prompt whose length sets the prefill size")
    p.add_argument("--json", type=str, default=None, help="write the sweep to this path")

    add_flops_args(p)
    add_model_args(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = CharTokenizer(DEFAULT_TEXT)

    if args.checkpoint is not None:
        import torch
        from generate import load_checkpoint

        model, tokenizer = load_checkpoint(args.checkpoint, torch.device("cpu"))
        config = model.config
    else:
        config = model_config_from_args(args, vocab_size=tokenizer.vocab_size)

    prompt_len = len(tokenizer.encode(args.prompt))
    _sanity_check(config, prompt_len)

    rows = sweep(config, prompt_len, args.seq_lens)

    G = 1e9
    print(f"Model: d_model={config.d_model} n_layers={config.n_layers} "
          f"ffn_dim={config.ffn_dim} vocab={config.vocab_size}")
    print(f"Matmul params: {param_count(config) / 1e6:.2f}M   "
          f"prompt_len={prompt_len}")
    print()
    print(f"{'new tokens':>10} {'naive GFLOPs':>14} {'cached GFLOPs':>14} {'naive/cached':>14}")
    for r in rows:
        print(f"{r['n_new']:>10} {r['naive_flops']/G:>14.2f} "
              f"{r['cached_flops']/G:>14.2f} {r['ratio']:>13.1f}x")
    print()
    print("Cached grows ~linearly with length; naive grows ~quadratically, "
          "so the ratio widens as you generate more.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "ffn_dim": config.ffn_dim,
            "vocab_size": config.vocab_size,
            "prompt_len": prompt_len,
            "matmul_params": param_count(config),
            "sweep": rows,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
