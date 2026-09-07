"""
Correctness/regression checks for MiniGPT.

There's no KV cache yet (see README), so this isn't a cached-vs-uncached
equivalence test — it checks the properties that must hold regardless:
the forward pass is deterministic in eval mode, and naive generation
produces the expected shape and a stable result for a fixed model/prompt.

    python test_equivalence.py
"""

from __future__ import annotations

import torch

from data import CharTokenizer, DEFAULT_TEXT
from generate import generate_naive
from model import GPTConfig, MiniGPT


def build_model(seed: int = 0) -> tuple[MiniGPT, CharTokenizer]:
    torch.manual_seed(seed)

    tokenizer = CharTokenizer(DEFAULT_TEXT)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=256,
        ffn_dim=128,
        dropout=0.0,
    )
    model = MiniGPT(config).eval()
    return model, tokenizer


def test_forward_is_deterministic():
    """With dropout=0 and eval mode, repeated forward passes on the same
    input must produce identical logits."""

    model, tokenizer = build_model()

    prompt = "the cat sat on"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(input_ids)
        logits_b = model(input_ids)

    assert torch.equal(logits_a, logits_b), "Forward pass is not deterministic"


def test_generate_naive_shape_and_determinism():
    model, tokenizer = build_model()

    prompt = "the cat"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    max_new_tokens = 50
    generated_a = generate_naive(model, input_ids, max_new_tokens=max_new_tokens)
    generated_b = generate_naive(model, input_ids, max_new_tokens=max_new_tokens)

    expected_len = input_ids.shape[1] + max_new_tokens
    assert generated_a.shape == (1, expected_len), (
        f"Expected shape (1, {expected_len}), got {tuple(generated_a.shape)}"
    )

    assert torch.equal(generated_a, generated_b), (
        "Greedy generation is not deterministic for repeated calls with the same prompt"
    )

    # The prompt itself must be preserved verbatim at the start of the output.
    assert torch.equal(generated_a[:, : input_ids.shape[1]], input_ids)


if __name__ == "__main__":
    test_forward_is_deterministic()
    test_generate_naive_shape_and_determinism()
    print("All tests passed.")
