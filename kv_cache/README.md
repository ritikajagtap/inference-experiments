# kv_cache

MiniGPT with a KV cache added to `CausalSelfAttention` / `MiniGPT`, so
generation doesn't have to reprocess the entire sequence at every step.
This is a sibling of `../mini-gpt/` (the naive baseline), not a
modification of it — diff the two `model.py` files to see exactly what a
KV cache changes:

```bash
diff ../mini-gpt/model.py model.py
```

Everything else (training, checkpoints, tokenizer) behaves identically
to the baseline; training a model here and training the same config in
`../mini-gpt/` should produce the same loss curve, since caching only
affects generation.

## How it works

`CausalSelfAttention.forward` takes an optional `cache = (past_k, past_v)`
and a `use_cache` flag:

- **Prefill**: the full prompt runs through the model once with
  `use_cache=True`. Each layer returns the `(k, v)` it computed.
- **Decode**: each later call passes in just the one new token, plus the
  caches from the previous step. Every layer concatenates its new `k, v`
  onto the cached ones before attending — so instead of O(T²) work to
  reprocess a growing sequence, each step does O(T) work (one new query
  attending to a cache that's grown by one).
- **RoPE** is applied using each token's *absolute* position
  (`start_pos` = however many tokens are already cached), so a token gets
  the same rotation whether it was processed during prefill or one at a
  time during decode.
- **Masking**: `is_causal=True` is only correct when the query and key
  lengths match (prefill, or any call with no cache) — that's the one
  case PyTorch's SDPA aligns as a standard lower-triangular mask. For a
  single-token decode step there's nothing to mask (one query, which is
  always the most recent token, may attend to the whole cache plus
  itself), so no mask is passed at all. For a multi-token continuation
  with a non-empty cache (query shorter than key, more than one new
  token), `CausalSelfAttention` builds an explicit offset boolean mask
  instead of relying on `is_causal`: on this torch build, `is_causal=True`
  with a shorter query aligns the triangle to the *upper-left* corner
  (query 0 would attend only to key 0) rather than the bottom-right
  corner a naive KV-cache implementation might assume — this was caught
  by a numerical check during development, not just the coarse
  cached-vs-uncached equivalence test, since that check alone wouldn't
  have exercised the multi-token-continuation code path.

`generate_cached` in `generate.py` implements the prefill-then-decode
loop; `generate_naive` (copied from the baseline) is kept alongside it so
`benchmark.py` and `test_equivalence.py` can compare the two directly.

## Files

Same layout as `../mini-gpt/`, plus the actual point of this folder:

- `test_equivalence.py` — proves the cache doesn't change behavior:
  cached and uncached generation produce identical tokens, prefill logits
  match a plain forward pass, and the cache grows by exactly one token
  per decode step.
- `benchmark.py` — the real cached-vs-uncached tok/s comparison (the
  version in `../mini-gpt/benchmark.py` only measures the naive baseline,
  since there was no cache to compare against yet).

## Usage

```bash
# Train (writes out/ckpt.pt and out/ckpt.loss.json)
python train.py --steps 2000 --checkpoint out/ckpt.pt

# Generate — cached by default, --no-cache to use the naive path
python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --max-new-tokens 100
python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --no-cache

# Benchmark cached vs uncached throughput
python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 200

# Also collect per-step latency + a length sweep, as JSON (for charting)
python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 200 --sweep --json out/bench.json

# Confirm the cache doesn't change outputs
python test_equivalence.py
```

## Measured result

With `out/ckpt.pt` trained on the toy corpus (`d_model=256, n_layers=6`),
on CPU, prompt `"the cat"`:

- Per-step decode latency: naive grows from ~9.5ms/step (steps 0-10) to
  ~25.8ms/step (steps 190-200) as the sequence lengthens; cached stays
  flat at ~4-5ms/step regardless of how long the sequence has grown.
- Throughput at 200 generated tokens: naive ~60 tok/s, cached ~214 tok/s
  — a **3.6x speedup**, and the gap widens as generation gets longer
  (2.0x at 25 tokens → 3.6x at 200).

Reproduce with `python benchmark.py --checkpoint out/ckpt.pt --sweep
--json out/bench.json`.

## What's not here yet

Peak memory usage (the cache trades compute for memory that grows with
context — not free) isn't measured yet. `--sweep`'s length sweep is also
only 4 points; a finer sweep would show the speedup-vs-length curve more
precisely.
