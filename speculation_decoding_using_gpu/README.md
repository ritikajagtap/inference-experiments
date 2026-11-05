# Speculative decoding on GPU

Make a large **target** model decode faster by letting a small **draft** model
guess the next few tokens, then checking all the guesses at once. Built with Hugging
Face Transformers and PyTorch, batch size one, on one CUDA GPU. This is a sibling of
[`../kv-cache-using-gpu/`](../kv-cache-using-gpu/): same careful setup (load the
models once; use the same weights, precision, inputs, attention kernel, and greedy
decoding every run; record the exact Hub commit), but a different speed trick.

The default pair is **Qwen/Qwen2.5-0.5B-Instruct** (draft) guessing for
**Qwen/Qwen2.5-7B-Instruct** (target). They are from the same family, so they share
a tokenizer and vocabulary — that is what lets the target check the draft's token
ids directly.

## How speculative decoding works

Normal decoding runs the target once per new token. At batch size one this is slow
because it is **memory-bound**: each pass re-reads all the model's weights just to
produce one token, so what really costs time is the number of passes you run one
after another. Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) cuts
that number down. Each round:

1. **Draft guesses.** The small model generates `gamma` tokens, one at a time. This
   is cheap because the draft is a fraction of the target's size.
2. **Target checks them in one pass.** The target scores all `gamma` guesses in a
   single forward pass. That one pass tells us, for each guessed token, what the
   target itself would have picked there, plus one extra **bonus** token for the
   position right after the last guess.
3. **Keep or fix.** Go through the guesses from the start, keeping each one that
   matches the target's own pick. At the first mismatch, use the target's pick
   instead (the **correction**) and throw the rest away. If every guess matched,
   also keep the bonus token.

So one target pass produces between **1** token (first guess already wrong) and
**`gamma + 1`** tokens (every guess right, plus the bonus). The target does the same
heavy math either way, but in far fewer passes — and the number of passes is what
makes batch-one decoding slow.

### Why the output is unchanged

Every token we keep is the target's own pick for that position, using the same
earlier tokens a normal decoder would have seen. So the text comes out **exactly the
same** as plain greedy decoding with the target alone — speculation only changes
speed. This is the same kind of guarantee the KV cache gives in
[`../kv_cache/`](../kv_cache/), and `test_speculative.py` checks it: even with a
draft that differs from the target (so guesses really do get rejected and
corrected), the output still matches the baseline token for token.

This folder does the **greedy** case, which is what the benchmark measures. Random
sampling needs a different keep-or-fix rule that keeps the target's output
distribution unchanged; it is mentioned here but not implemented.

### KV caches and undoing rejected guesses

Both models keep a `DynamicCache`. Each round, a model is fed only the tokens it has
not seen yet, then the new ones. After the round, both caches are trimmed back to the
committed length, which drops the keys and values for rejected guesses (and, in the
target, for any position whose token was replaced by a correction). Accepted tokens
keep their cache entries. The one tricky bit: after a round where every guess was
accepted, the draft never fed in its last guess or the bonus token, so those few
tokens are simply fed again at the start of the next round. See the comments in
`speculative_decoding.py`.

## Files

- `speculative_decoding.py` — the algorithm. `speculative_generate` (draft propose
  + target verify + accept/correct, with cache rollback) and `greedy_generate` (the
  target-only baseline), sharing one KV-cache forward helper, plus a `SpecStats`
  accounting record. Pure model code — no I/O, no CUDA assumptions.
- `models.py` — load a draft/target pair onto the GPU and the shared tokenizer;
  resolves immutable Hub commits and asserts a shared vocabulary.
- `generate.py` — CLI: run speculative decoding on a prompt, print the text and the
  acceptance rate, and (by default) compare against the baseline.
- `benchmark.py` — the real speculative-vs-baseline throughput benchmark; writes a
  CSV. Sweeps prompt lengths and `gamma`.
- `plot_results.py` — median throughput, speedup, acceptance, and memory plots from
  one complete CSV.
- `record_trace.py` — run the decode on a GPU and write `trace.json`: the per-round
  structure plus real GPU utilization sampled from the driver (NVML). Feeds the monitor.
- `monitor.html` — an interactive page that replays a run and shows the draft keeping
  the GPU busy while its compute (SM) stays low. See "Watching the GPU compute" below.
- `test_speculative.py` — offline tests (tiny random Qwen2 models on CPU): the greedy
  equivalence guarantee, the serial-pass win, accounting, and CLI/plot validation.

## Hardware and memory

Use an NVIDIA CUDA GPU with a CUDA-enabled PyTorch build. BF16 is the default; use
`--precision float16` on GPUs without BF16. There is no CPU fallback. Speculative
decoding keeps **two** models resident at once: the 7B target plus the 0.5B draft
(≈ 14.2 + 1.0 GiB of BF16 weights before runtime buffers and the two KV caches), so
budget more memory than a target-only run. An out-of-memory error suggests a larger
GPU, freeing other workloads, shorter lengths, a smaller `gamma`, or FP16.

## Setup on an existing rented GPU (for example RunPod)

Manual instructions for a GPU machine you already control; the scripts do not rent
instances or make cloud API calls. Choose a Linux PyTorch/CUDA image, upload this
directory, and use a persistent volume for the Hugging Face download cache.

```bash
nvidia-smi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python -m pip freeze > environment.txt
```

If the image lacks CUDA-enabled PyTorch, install the wheel selected by the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before the
requirements; a CPU build will not work. On Windows, activate with
`.\.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

The first run downloads both models (≈ 15 GB for the 7B target plus ≈ 1 GB for the
draft). Set `HF_HOME` to a persistent cache directory if desired. Download and model
load are excluded from measurements.

## Run

Quick human-readable demo (speculative vs baseline, with acceptance rate):

```bash
python generate.py --prompt "Explain speculative decoding in one sentence." --max-new-tokens 128
python generate.py --prompt "..." --gamma 6 --chat          # wrap in the chat template
```

Smoke test (still downloads both models):

```bash
python benchmark.py --prompt-lengths 128 --generated-tokens 8 --gammas 4 \
    --warmups 1 --repeats 1 --csv results/smoke.csv
```

Fuller benchmark — two prompt lengths and a `gamma` sweep:

```bash
python benchmark.py --prompt-lengths 512 2048 --generated-tokens 128 \
    --gammas 2 4 6 8 --warmups 2 --repeats 5 --precision bfloat16 --csv results/full.csv

python plot_results.py results/full.csv --output-dir results/plots
```

`--draft-revision` / `--target-revision` pin Hub commits to reproduce an earlier
CSV. Other options: `--draft`, `--target`, `--device`, `--seed`, `--precision`, and
the custom-prompt inputs below. Existing CSVs are never overwritten.

### Give your own prompt

```bash
python benchmark.py --prompt "Explain KV caching with an example." --csv results/custom.csv
python benchmark.py --prompt-file prompt.txt --csv results/from_file.csv
python benchmark.py --interactive --csv results/interactive.csv
```

These three are mutually exclusive. Custom input defaults to **natural** mode
(tokenize the prompt once, benchmark that exact sequence at its own length);
`--prompt-lengths` applies only in **exact** mode (repeat/truncate the tokenized text
to each length). With no custom input, the built-in passage and exact-length defaults
apply. The CSV stores the prompt text and token ids, so avoid sharing CSVs built from
private prompts.

## What is measured

Each run makes exactly `--generated-tokens` new tokens and ignores EOS, so the
baseline and every `gamma` produce the same text. The speculative output is checked
against the baseline, and any mismatch stops the run. The timer covers the whole
generation, including the first pass over the prompt for both models, with the GPU
synchronized before the timer starts and before it stops. The main number is
end-to-end **output tokens per second**. It is not decode-only speed, and it is not
a production server number.

The more telling numbers are the **acceptance rate** (the share of the draft's
guesses the target kept) and the **tokens committed per target pass** (always 1 for
the baseline; up to `gamma + 1` for speculation). Peak allocated memory is the
highest amount of GPU memory PyTorch held for tensors during the run — including the
weights of **both** models — not the total device usage you would see in
`nvidia-smi`.

How big the speedup is depends mostly on how well the draft predicts the target. A
high acceptance rate means each target pass moves several tokens forward. A larger
`gamma` can win more per pass, but it also wastes more draft work when the guesses
are often wrong, so the best `gamma` is usually a middle value. Only compare runs
that used the same settings and whose outputs all matched.

## Watching the GPU compute

The benchmark gives you numbers; `monitor.html` lets you *watch* a run and see why
the draft is cheap in compute even though it runs many times. At batch size one,
every pass is spent mostly re-reading weights from memory, so the GPU's compute (SM)
units stay mostly idle. The draft makes this clear: it runs several tiny passes per
round that keep the GPU busy but barely use its math units; the target's one batched
check uses them a little more.

You do **not** record your screen. The GPU run writes a small `trace.json`, and the
page replays it — interactive, with the real measured numbers:

1. On the GPU, record a run (this also samples real GPU utilization via NVML):

   ```bash
   pip install nvidia-ml-py
   python record_trace.py --prompt "Explain speculative decoding." --gamma 4 \
       --max-new-tokens 96 --out trace.json
   ```

2. Download `trace.json` to your machine (a few KB).
3. Open `monitor.html` (locally, or the published page) and click **Load trace.json**.
   Play, pause, scrub the timeline, and read the per-phase compute and memory meters —
   now showing the SM % you actually measured.

The page ships with a built-in *modeled* sample so it works before you have a GPU run;
the badge at the top says `modeled` until you load a measured trace, then `measured`.

## CSV columns

| Columns | Meaning |
| --- | --- |
| `experiment_id`, `timestamp_utc` | Experiment UUID and measured-row completion time |
| `draft_model`, `target_model`, `requested_*_revision`, `draft_revision`, `target_revision` | Hub IDs, requested refs, resolved commits |
| `torch_version`, `transformers_version`, `python_version`, `cuda_version`, `package_versions` | Runtime/library metadata; versions are JSON |
| `gpu`, `gpu_total_mib`, `device`, `precision` | Device identity, capacity, visible index, dtype |
| `attention_implementation`, `decoding`, `batch_size`, `seed`, `deterministic_algorithms`, `tf32` | Fixed inference settings |
| `warmups`, `repeats`, `generated_tokens`, `prompt_lengths`, `gammas` | Experiment settings; lengths and gammas are JSON |
| `prompt_tokens`, `prompt_sha256`, `prompt_source`, `prompt_mode`, `prompt_text`, `input_token_ids` | Exact workload: count, hash, source, mode, text, ids |
| `method`, `gamma`, `repeat` | `baseline` or `speculative`; draft length (blank for baseline); 1-based repeat |
| `output_token_ids`, `tokens_match` | Generated ids as JSON and equality to the baseline reference |
| `total_seconds`, `output_tokens_per_second` | End-to-end timing including both models' prefill |
| `rounds`, `target_forwards`, `draft_forwards`, `committed` | Serial-pass accounting (baseline: one target forward per token) |
| `acceptance_rate`, `mean_accepted_per_round`, `mean_committed_per_round` | Draft quality and tokens emitted per target forward |
| `baseline_allocated_mib`, `peak_allocated_mib`, `peak_reserved_mib` | Per-process PyTorch GPU memory, in MiB |

## Offline verification

No GPU benchmark results are supplied. Local checks run without downloading weights:

```bash
python -m py_compile speculative_decoding.py models.py generate.py benchmark.py plot_results.py test_speculative.py
python -m unittest -v test_speculative.py
```

The tests use tiny randomly initialized Qwen2 models on CPU and cover: the greedy
equivalence guarantee (speculative output equals the baseline, with a differing draft
so rejection/correction actually runs), the serial-pass reduction when the draft is
the target, `SpecStats` math, the `gamma >= 1` guard, benchmark CLI validation, and
plotting (PNG/medians from a synthetic fixture, and rejection of an incomplete CSV).
This is a correctness check, not a Qwen2.5 performance measurement.

Verified on 2026-10-27: all six source files compiled; all seven offline tests passed
(Python 3.10.4, PyTorch 2.14.0+cpu, Transformers 4.57.6). PyTorch reported
`cuda.is_available() == False`; `benchmark.py` and `generate.py` were invoked and
exited immediately with the CUDA-unavailable message, before downloading or loading
any weights. No CUDA inference, model loading, or GPU performance/memory numbers were
verified.
```bash
python benchmark.py --prompt-lengths 128 --generated-tokens 8 --gammas 4 --warmups 1 --repeats 1 --csv results/smoke.csv
# -> CUDA is unavailable. Install CUDA-enabled PyTorch and use a CUDA GPU; CPU fallback is disabled.
```
