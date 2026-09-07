"""
MiniGPT model definitions: RMSNorm, RoPE, SwiGLU, causal self-attention.

Ported directly from the 01 notebook. Pure `nn.Module` code — no I/O, no
training loop, no argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 2048
    ffn_dim: int = 768
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.weight * x


def precompute_rope_freqs(
    seq_len: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cosine and sine values used by RoPE."""

    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    positions = torch.arange(seq_len, device=device).float()

    freqs = torch.outer(positions, theta)

    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embeddings to Q and K.

    q, k shape: (batch, heads, seq_len, head_dim)
    """

    q1 = q[..., 0::2]
    q2 = q[..., 1::2]

    k1 = k[..., 0::2]
    k2 = k[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated_1 = q1 * cos - q2 * sin
    q_rotated_2 = q1 * sin + q2 * cos

    k_rotated_1 = k1 * cos - k2 * sin
    k_rotated_2 = k1 * sin + k2 * cos

    q_out = torch.stack([q_rotated_1, q_rotated_2], dim=-1).flatten(-2)
    k_out = torch.stack([k_rotated_1, k_rotated_2], dim=-1).flatten(-2)

    return q_out, k_out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.max_seq_len = config.max_seq_len

        self.head_dim = self.d_model // self.n_heads

        assert self.d_model % self.n_heads == 0

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        cos, sin: RoPE tables for this sequence length/head_dim, precomputed
        once per forward pass by MiniGPT and shared across every layer
        (they're identical layer to layer since head_dim is fixed model-wide).
        """

        B, T, C = x.shape

        # 1. Create Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Split into attention heads
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        # 4-7. Scaled dot-product attention with a fused/flash kernel where
        # available. `is_causal=True` applies the causal mask internally —
        # no need to materialize a (T, T) mask tensor ourselves.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # 8. Combine heads
        out = out.transpose(1, 2)
        out = out.contiguous()
        out = out.view(B, T, self.d_model)

        # 9. Output projection
        out = self.out_proj(out)

        return out


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        d_model = config.d_model
        ffn_dim = config.ffn_dim

        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = F.silu(gate)

        up = self.up_proj(x)

        x = gate * up
        x = self.down_proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.head_dim = config.d_model // config.n_heads

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = RMSNorm(self.d_model)

        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)

        B, T = input_ids.shape

        assert T <= self.max_seq_len

        # 1. Token embedding: (B, T) -> (B, T, d_model)
        x = self.token_embedding(input_ids)

        # RoPE tables are identical for every layer (head_dim is fixed
        # model-wide), so compute them once per forward pass instead of
        # redundantly inside each layer's attention.
        cos, sin = precompute_rope_freqs(T, self.head_dim, x.device)

        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        # 3. Final normalization
        x = self.final_norm(x)

        # 4. Language model head: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits
