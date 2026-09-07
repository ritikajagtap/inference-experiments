"""
Verify the KV cache is a pure speed optimization: cached and uncached
generation must produce bit-identical token sequences (and near-identical
logits, modulo floating-point summation order) for the same model/prompt.

    python test_equivalence.py
"""

from __future__ import annotations

import torch

from data import CharTokenizer, DEFAULT_TEXT
from generate import generate_cached, generate_naive
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


def test_generation_tokens_match():
    model, tokenizer = build_model()

    prompt = "the cat"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    cached = generate_cached(model, input_ids, max_new_tokens=50)
    uncached = generate_naive(model, input_ids, max_new_tokens=50)

    assert torch.equal(cached, uncached), (
        f"Cached and uncached generations diverged:\n"
        f"cached  : {tokenizer.decode(cached[0].tolist())!r}\n"
        f"uncached: {tokenizer.decode(uncached[0].tolist())!r}"
    )


def test_prefill_logits_match_plain_forward():
    """Prefill-with-cache logits must match a plain forward pass over the same tokens."""

    model, tokenizer = build_model()

    prompt = "the cat sat on"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        logits_plain, _ = model(input_ids)
        logits_cached, caches = model(input_ids, use_cache=True)

    assert torch.allclose(logits_plain, logits_cached, atol=1e-5), (
        "Prefill logits differ between cached and uncached forward passes"
    )

    # Extend by one token both ways and check the next-token logits agree.
    next_token = torch.argmax(logits_plain[:, -1, :], dim=-1, keepdim=True)
    extended = torch.cat([input_ids, next_token], dim=1)

    with torch.no_grad():
        logits_plain_next, _ = model(extended)
        logits_cached_next, _ = model(next_token, caches=caches, use_cache=True)

    assert torch.allclose(
        logits_plain_next[:, -1, :], logits_cached_next[:, -1, :], atol=1e-5
    ), "Next-token logits differ between cached and uncached forward passes"


def test_cache_grows_by_one_token_per_decode_step():
    model, tokenizer = build_model()

    input_ids = torch.tensor([tokenizer.encode("the cat")], dtype=torch.long)

    with torch.no_grad():
        _, caches = model(input_ids, use_cache=True)
        prompt_len = input_ids.shape[1]
        assert caches[0][0].shape[2] == prompt_len

        next_token = torch.zeros((1, 1), dtype=torch.long)
        _, caches = model(next_token, caches=caches, use_cache=True)
        assert caches[0][0].shape[2] == prompt_len + 1


if __name__ == "__main__":
    test_generation_tokens_match()
    test_prefill_logits_match_plain_forward()
    test_cache_grows_by_one_token_per_decode_step()
    print("All tests passed.")
