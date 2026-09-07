"""
Character-level tokenizer and batching for MiniGPT.

Same toy corpus as the 01 notebook by default; pass --data-file to
train.py to point at a real text file instead.
"""

from __future__ import annotations

import torch

DEFAULT_TEXT = """
the cat sat on the mat.
the cat ate the food.
the dog sat on the mat.
the dog ate the food.
the cat and the dog are friends.
the cat likes the dog.
the dog likes the cat.
"""


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[i] for i in tokens)

    def state_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.chars = state["chars"]
        tok.vocab_size = len(tok.chars)
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        return tok


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Input sequences
    x = torch.stack([data[i : i + block_size] for i in ix])

    # Target sequences
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
