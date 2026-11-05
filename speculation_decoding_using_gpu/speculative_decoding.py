"""Greedy speculative decoding, plus the plain greedy baseline it is checked against.

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) makes a large
"target" model decode faster by letting a small "draft" model guess the next few
tokens, then checking all the guesses in one target pass. Each round:

1. The draft proposes `gamma` tokens, one at a time (greedy).
2. The target scores all `gamma` proposals in a SINGLE forward pass. That pass
   tells us, for each proposed token, what the target itself would have picked
   there, plus one extra "bonus" token for the position after the last proposal.
3. Keep the proposals from the start that match the target's own pick. At the
   first one that does not match, use the target's pick instead (the "correction")
   and drop the rest. If every proposal matched, also keep the bonus token.

So one target pass commits between 1 and `gamma + 1` tokens. The target still does
the same heavy matmuls, but in far fewer passes run back to back — and that pass
count is what makes batch-one decoding slow, because it is memory-bound.

The output is exactly what plain greedy decoding would produce: every token we keep
is the target's own greedy pick for that position, given the same earlier tokens a
plain decoder would have seen. Speculation changes only speed, not the text — the
same kind of guarantee the KV cache gives in `../kv_cache/`, and `test_speculative.py`
checks it. (Random sampling needs a different accept/reject rule; this file does the
greedy case, which is what the benchmark measures.)

Plain model code: no argument parsing, no file I/O, no CUDA assumptions. It runs
wherever the models and tensors already live.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass
class SpecStats:
    """Counts from one generation: used to report speed and how well the draft did."""

    proposed: int = 0          # draft tokens offered for checking
    accepted: int = 0          # draft tokens the target kept (not counting corrections/bonus)
    committed: int = 0         # total new tokens emitted
    rounds: int = 0            # propose-then-check cycles
    target_forwards: int = 0   # target passes run back to back — the cost that matters
    draft_forwards: int = 0    # draft passes run back to back

    @property
    def acceptance_rate(self) -> float:
        """Share of proposed tokens that were kept. Higher means the draft is a good
        stand-in for the target, so each target pass moves further ahead."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0

    @property
    def mean_committed_per_round(self) -> float:
        """New tokens per target pass. A plain decoder gets exactly 1; this is the speedup."""
        return self.committed / self.rounds if self.rounds else 0.0


def _forward(
    model: torch.nn.Module, cache: DynamicCache, tokens: torch.Tensor, keep: int
) -> tuple[torch.Tensor, DynamicCache]:
    """Run `model` on `tokens`, adding them to `cache`.

    `cache` must already hold the tokens right before `tokens`; callers keep this
    true by feeding each model only the tokens it has not seen yet. Returns the
    logits for the last `keep` positions, shape (1, keep, vocab), and the grown cache.
    """
    attn_mask = torch.ones(
        (1, cache.get_seq_length() + tokens.shape[1]), dtype=torch.long, device=tokens.device
    )
    out = model(input_ids=tokens, attention_mask=attn_mask, past_key_values=cache,
                use_cache=True, logits_to_keep=keep)
    return out.logits, out.past_key_values


def _greedy_token(logits: torch.Tensor) -> torch.Tensor:
    """The greedy next token from the last position's logits, shape (1, 1)."""
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _tokens_not_cached(prefix: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    """The committed tokens a model has not fed into its cache yet."""
    return prefix[:, cache.get_seq_length():]


def _crop(cache: DynamicCache, length: int) -> None:
    """Shrink `cache` to `length` tokens (does nothing if it is already that short)."""
    if cache.get_seq_length() > length:
        cache.crop(length)


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None = None,
    vocab_limit: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Plain greedy decoding with a KV cache: one target pass per token. This is the
    baseline every speculative run is compared against. Returns (sequence, stats);
    stats fills in only the fields that apply, so both generators report the same way.

    `vocab_limit` restricts the argmax to the first N logits. Pass the draft/target
    shared vocab when using this as the reference for a speculative run, so both see
    the same effective vocabulary (model families often pad the lm_head differently).
    """
    cache = DynamicCache()
    stats = SpecStats()
    generated: list[torch.Tensor] = []
    next_token = None

    for step in range(max_new_tokens):
        feed = input_ids if step == 0 else next_token
        logits, cache = _forward(model, cache, feed, keep=1)
        stats.target_forwards += 1
        stats.committed += 1
        next_token = _greedy_token(logits if vocab_limit is None else logits[..., :vocab_limit])
        generated.append(next_token)
        if eos_id is not None and next_token.item() == eos_id:
            break

    stats.rounds = stats.target_forwards
    return torch.cat([input_ids, *generated], dim=1), stats


@torch.no_grad()
def speculative_generate(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Greedy speculative decoding: the draft proposes `gamma` tokens each round and
    the target checks them all in one pass.

    `draft` and `target` must share a tokenizer so the target can check the draft's
    token ids directly. Returns (sequence, stats). The sequence is cut to exactly
    `max_new_tokens` new tokens (a round can run a little past that).

    Both models' argmax is restricted to the shared vocabulary (the smaller of the
    two lm_head sizes). Model families often pad the lm_head to different sizes — e.g.
    Qwen2.5-0.5B has 151936 rows and Qwen2.5-7B has 152064 — but the extra rows are
    unused padding, so clipping to the smaller size keeps every token id valid for
    both models and loses no real token.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")

    vocab_limit = min(draft.config.vocab_size, target.config.vocab_size)
    prompt_len = input_ids.shape[1]
    draft_cache = DynamicCache()
    target_cache = DynamicCache()
    prefix = input_ids                       # the prompt plus everything committed so far
    stats = SpecStats()

    while prefix.shape[1] - prompt_len < max_new_tokens:
        # 1. Draft proposes `gamma` tokens, one at a time. First feed it the committed
        #    tokens it has not seen yet, then feed each new proposal straight back in.
        proposals: list[torch.Tensor] = []
        feed = _tokens_not_cached(prefix, draft_cache)
        for _ in range(gamma):
            logits, draft_cache = _forward(draft, draft_cache, feed, keep=1)
            stats.draft_forwards += 1
            feed = _greedy_token(logits[..., :vocab_limit])
            proposals.append(feed)
        proposal_ids = torch.cat(proposals, dim=1)        # (1, gamma)

        # 2. Target checks every proposal in one pass. Feed it the tokens it has not
        #    seen yet plus all the proposals. The last gamma+1 logits give the target's
        #    own pick for each proposal, then one bonus pick after the last proposal.
        target_feed = torch.cat([_tokens_not_cached(prefix, target_cache), proposal_ids], dim=1)
        logits, target_cache = _forward(target, target_cache, target_feed, keep=gamma + 1)
        stats.target_forwards += 1
        target_picks = logits[0, :, :vocab_limit].argmax(dim=-1)   # (gamma+1,) one pick per position

        # 3. Keep proposals from the start for as long as they match the target's pick.
        stats.rounds += 1
        stats.proposed += gamma
        n_accept = 0
        while n_accept < gamma and proposal_ids[0, n_accept].item() == target_picks[n_accept].item():
            n_accept += 1
        stats.accepted += n_accept

        # Commit the accepted proposals plus one more token — the target's own pick at
        # the stopping point. If all gamma matched, that extra token is the bonus;
        # otherwise it is the correction replacing the first wrong proposal. Both cases
        # are the same slice because target_picks has gamma+1 entries.
        correction = target_picks[n_accept].view(1, 1)
        new_tokens = torch.cat([proposal_ids[:, :n_accept], correction], dim=1)
        prefix = torch.cat([prefix, new_tokens], dim=1)
        stats.committed += new_tokens.shape[1]

        # Drop rejected tokens from both caches so the next round feeds the right
        # tokens. Accepted tokens keep their valid cache entries: keep everything up to
        # prefix[:-1] and re-feed the last token (and any gap) next round.
        keep_len = prefix.shape[1] - 1
        _crop(target_cache, keep_len)
        _crop(draft_cache, keep_len)

        # Stop at the first EOS among the tokens we just committed.
        if eos_id is not None:
            hits = (new_tokens[0] == eos_id).nonzero()
            if hits.numel():
                new_start = prefix.shape[1] - new_tokens.shape[1]
                prefix = prefix[:, : new_start + int(hits[0]) + 1]
                break

    return prefix[:, : prompt_len + max_new_tokens], stats
