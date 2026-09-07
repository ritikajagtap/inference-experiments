# mini-gpt

MiniGPT (RMSNorm + RoPE + SwiGLU + causal self-attention) lifted out of the
`mini_gpt.ipynb` notebook into a proper module layout. Model behavior
(loss curve, generated text) is unchanged from the notebook. **There is no
KV cache in this folder** — `generate.py` uses naive generation,
reprocessing the whole sequence through every layer for each new token.
This is the baseline to compare against.

The KV-cached version lives in a sibling folder, `../kv_cache/` (its own
`model.py`, `train.py`, etc.) rather than as a flag in these files — see
`../kv_cache/README.md` for how the cache works and
`../kv_cache/benchmark.py` for the actual cached-vs-uncached speed
comparison.

One deviation from the notebook: attention uses
`F.scaled_dot_product_attention` (fused/flash-attention kernel where
available, causal masking handled internally) instead of manually
computing `QK^T` → mask → softmax → `V`, and RoPE's cos/sin tables are
computed once per forward pass in `MiniGPT` and shared across all 6
layers instead of being recomputed independently inside each layer. Both
are implementation-efficiency fixes, not behavior changes — they remove
work that was identical across layers and calls, which matters here
because it's exactly the kind of overhead a KV-cache "before" baseline
needs to be free of for a fair before/after comparison later.

## Files

- `config.py` — **the one file to edit to change hyperparameters.** Holds
  `DEFAULTS` (every model/training/generation/benchmark default value) and
  the shared argparse builders (`add_model_args`, `add_train_args`,
  `add_generation_args`, `add_benchmark_args`) that `train.py`,
  `generate.py`, and `benchmark.py` all use — so a default only needs to
  change in one place, and every script still accepts `--flag value`
  overrides on top of it.
- `model.py` — `nn.Module` definitions (`RMSNorm`, RoPE helpers, `SwiGLU`,
  `CausalSelfAttention`, `TransformerBlock`, `MiniGPT`) and the `GPTConfig`
  dataclass. Pure — no I/O, no training logic, no hyperparameter defaults
  (those live in `config.py`). Ported from notebook cells 3, 5, 6, 8, 10-12.
- `data.py` — character-level tokenizer (`CharTokenizer`) and `get_batch`.
  Uses the same toy corpus as the notebook by default. Ported from cells
  13-15.
- `train.py` — training loop, AdamW optimizer, checkpointing, logging.
  Ported from cells 17-26.
- `generate.py` — loads a checkpoint and greedily samples from it via
  `generate_naive`, the same full-recompute decoding as the notebook.
  Ported from cells 32-34.
- `benchmark.py` — measures naive-generation tokens/sec, as a baseline to
  compare against once a KV cache exists.
- `test_equivalence.py` — checks the forward pass is deterministic and
  that `generate_naive` produces the right shape/prompt-preservation.
  Since there's no cache yet, this checks determinism rather than
  cached-vs-uncached equivalence — it'll be extended to that once a
  cache exists.
- `modal_app.py` — runs `train.py` on a Modal GPU. See "GPU / Modal" below.

## Seeing the loss curve

`train.py` writes the full per-step loss history as JSON alongside the
checkpoint — `out/ckpt.pt` → `out/ckpt.loss.json` (a flat array, one float
per step). Plot it however's convenient (matplotlib, a notebook, a
spreadsheet); there's no built-in plotting script in this repo.

## Changing hyperparameters

Edit the `Defaults` dataclass in `config.py` to change what every script
uses when you don't pass a flag — e.g. bump `d_model` or `n_layers` there
and `train.py`/`benchmark.py` both pick it up. For a one-off run, pass the
CLI flag instead (`python train.py --d-model 512`); flags always win over
`config.py`'s defaults.

## Usage

```bash
# Train (writes out/ckpt.pt and out/ckpt.loss.json)
python train.py --steps 2000 --checkpoint out/ckpt.pt

# Generate
python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --max-new-tokens 100

# Benchmark naive-generation throughput
python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 200

# Run correctness checks
python test_equivalence.py
```

## Testing

There's one automated check plus a manual end-to-end smoke test — run
both after changing `model.py`, `data.py`, or `train.py`.

**1. Regression test:**

```bash
python test_equivalence.py
```

Passes silently with `All tests passed.`; it checks that the forward pass
is deterministic and that `generate_naive` preserves the prompt and
produces the right output shape. If this fails after an edit, something
in the forward pass changed in a way that breaks reproducibility.

**2. Manual smoke test** (confirms training actually learns and generation
produces sane text — `test_equivalence.py` alone can't catch a broken
loss function or corrupted checkpoint):

```bash
# Short run — loss should drop from ~3.0 to well under 0.5 within 100-200 steps
python train.py --steps 200 --log-every 50 --checkpoint out/ckpt.pt

# Output should look like the training corpus (real words, not gibberish)
python generate.py --checkpoint out/ckpt.pt --prompt "the cat" --max-new-tokens 60

# Should run without error and print a tok/s number
python benchmark.py --checkpoint out/ckpt.pt --max-new-tokens 50
```

If you don't have a GPU locally, all three scripts fall back to CPU
automatically (`torch.cuda.is_available()` check) — the toy corpus is
small enough that CPU training is still fast.

## GPU / Modal

`train.py` exposes a plain `train(args)` function with no Modal
decorators, so it runs identically locally or wrapped for a remote GPU.
`modal_app.py` does that wrapping.

**One-time account setup:**

```bash
pip install modal
modal setup       # opens a browser to link this machine to your Modal account
```

`modal setup` writes `~/.modal.toml` with your auth token — it's
interactive (browser login), so it needs to be run by you once, not
scriptable from here.

**Running training on a GPU:**

```bash
modal run modal_app.py --steps 2000
modal run modal_app.py --steps 2000 --d-model 512 --n-layers 8
```

This builds a container image (Python 3.11 + `torch`, no local CUDA
needed on your machine), runs `train()` from `train.py` on a T4 GPU,
saves the checkpoint to a persistent Modal Volume (`mini-gpt-checkpoints`,
survives across runs), and downloads it to `./out/` when the run
finishes. CLI flags mirror `train.py`'s (`--steps`, `--d-model`, `--lr`,
etc.), with defaults pulled from the same `config.py` used locally.

To point `generate.py` or `benchmark.py` at a GPU-trained checkpoint,
just run them against the downloaded file — no Modal involvement needed
for inference on this toy model:

```bash
python generate.py --checkpoint out/ckpt.pt --prompt "the cat"
```
