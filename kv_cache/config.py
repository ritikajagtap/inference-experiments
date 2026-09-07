"""
Single source of truth for MiniGPT's hyperparameters and CLI flags.

To change a default (model size, learning rate, generation length, ...)
edit `DEFAULTS` below — train.py, generate.py, and benchmark.py all read
from here instead of each declaring their own argparse defaults, so a
change in one place takes effect everywhere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import GPTConfig


@dataclass(frozen=True)
class Defaults:
    # --- Model architecture (see model.GPTConfig) ---
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 2048
    ffn_dim: int = 768
    dropout: float = 0.0

    # --- Training ---
    batch_size: int = 4
    block_size: int = 32
    lr: float = 3e-4
    steps: int = 2000
    log_every: int = 100
    seed: int = 0
    checkpoint: str = "out/ckpt.pt"

    # --- Generation ---
    prompt: str = "the cat"
    max_new_tokens: int = 100

    # --- Benchmarking (a longer run than generate.py's default, for a
    # more stable tokens/sec measurement) ---
    benchmark_new_tokens: int = 200
    benchmark_warmup_tokens: int = 20


DEFAULTS = Defaults()


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Architecture flags — used by train.py, and by benchmark.py when it
    builds an untrained model instead of loading a checkpoint."""

    parser.add_argument("--d-model", type=int, default=DEFAULTS.d_model)
    parser.add_argument("--n-heads", type=int, default=DEFAULTS.n_heads)
    parser.add_argument("--n-layers", type=int, default=DEFAULTS.n_layers)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULTS.max_seq_len)
    parser.add_argument("--ffn-dim", type=int, default=DEFAULTS.ffn_dim)
    parser.add_argument("--dropout", type=float, default=DEFAULTS.dropout)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--block-size", type=int, default=DEFAULTS.block_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    parser.add_argument("--steps", type=int, default=DEFAULTS.steps)
    parser.add_argument("--log-every", type=int, default=DEFAULTS.log_every)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", type=str, default=DEFAULTS.prompt)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS.max_new_tokens)


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", type=str, default=DEFAULTS.prompt)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS.benchmark_new_tokens)
    parser.add_argument("--warmup-tokens", type=int, default=DEFAULTS.benchmark_warmup_tokens)


def model_config_from_args(args: argparse.Namespace, vocab_size: int) -> "GPTConfig":
    """Build a GPTConfig from parsed args that included add_model_args().

    Imports model.GPTConfig lazily so `import config` alone (e.g. from
    modal_app.py's local entrypoint) doesn't require torch to be
    installed on the machine invoking `modal run`.
    """
    from model import GPTConfig

    return GPTConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
