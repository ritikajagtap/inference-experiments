"""
MiniGPT model definitions: RMSNorm, RoPE, SwiGLU, causal self-attention —
with KV caching in `CausalSelfAttention` / `MiniGPT`.

This is the ../model.py baseline plus cache plumbing; diff the two files
to see exactly what a KV cache changes. Pure `nn.Module` code — no I/O,
no training loop, no argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


# One layer's cache: (past_k, past_v), each (B, n_heads, T_past, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]


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
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for absolute positions [start_pos, start_pos + seq_len).

    `start_pos` is the number of tokens already in the KV cache — a token
    must get the same rotation whether it's processed during prefill or
    one at a time during cached decoding, so RoPE always needs each
    token's *absolute* position, not its position within the current call.
    """

    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()

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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        x: (B, T, C) — T is the number of *new* tokens in this call (the
        full prompt on prefill, then one token at a time during decode).

        cache, if given, holds every (k, v) computed so far for this
        layer. New keys/values are appended to it before attending, so a
        decode call only pays for the new token's projection + attention
        work instead of recomputing the whole sequence.
        """

        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # `is_causal=True` only gives the right mask when q and k are the
        # same length (prefill, or any call with no cache): it aligns the
        # causal triangle to the *upper-left* corner, so with a shorter q
        # (decode, T < T_total) it would wrongly let query i attend only
        # to keys 0..i instead of the whole cache. Build an explicit
        # offset mask whenever queries and keys don't line up 1:1.
        T_total = k.shape[2]
        if T == T_total:
            attn_mask = None
            is_causal = True
        elif T == 1:
            # A single new query is always the most recent token, so it
            # may attend to the entire cache plus itself — no mask needed.
            attn_mask = None
            is_causal = False
        else:
            start_pos = T_total - T
            query_pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
            key_pos = torch.arange(T_total, device=x.device).unsqueeze(0)
            attn_mask = key_pos <= query_pos  # (T, T_total) bool, True = attend
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        return out, new_cache


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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, new_cache = self.attention(
            self.attn_norm(x), cos, sin, cache=cache, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


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

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: Optional[list[Optional[KVCache]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[KVCache]]]:
        """
        input_ids: (B, T) — T new tokens (the full prompt on the first
            call, then one token at a time once `caches` is populated).

        Returns (logits, new_caches). `new_caches` is a list with one
        (k, v) entry per layer when use_cache=True, else None.
        """

        B, T = input_ids.shape

        if caches is None:
            caches = [None] * len(self.blocks)

        start_pos = caches[0][0].shape[2] if caches[0] is not None else 0
        assert start_pos + T <= self.max_seq_len

        x = self.token_embedding(input_ids)

        # RoPE tables are identical for every layer (head_dim is fixed
        # model-wide), so compute them once per forward pass.
        cos, sin = precompute_rope_freqs(T, self.head_dim, x.device, start_pos=start_pos)

        new_caches = [] if use_cache else None

        for block, layer_cache in zip(self.blocks, caches):
            x, layer_new_cache = block(x, cos, sin, cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(layer_new_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_caches
