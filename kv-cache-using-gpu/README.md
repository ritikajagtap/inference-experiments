# KV-cache inference benchmark

Compare **Qwen/Qwen2.5-7B-Instruct** with KV caching ON and OFF using Hugging Face Transformers and PyTorch, batch size one, on one CUDA GPU. The model is loaded once; every pair uses the same GPU, weights, precision, exact input IDs, output length, SDPA attention, and greedy argmax decoding. Defaults: prompts of 128, 512, 1024, and 2048 tokens; 64 output tokens; two warmups per length/configuration; five measured pairs per length.

## Model and official references

The [official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) describes an instruction-tuned causal language model with 7.61 billion parameters, 28 layers, and grouped-query attention (28 query heads, four KV heads). Its default configuration supports 32,768 positions; the advertised 131,072 context requires long-context configuration changes. This project does not enable YaRN or change that configuration.

APIs were selected after consulting the official [Transformers cache guide](https://huggingface.co/docs/transformers/kv_cache), [Qwen2 forward API](https://huggingface.co/docs/transformers/model_doc/qwen2), [PyTorch CUDA timing and memory notes](https://docs.pytorch.org/docs/stable/notes/cuda), and [CUDA synchronization API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html). Requirements constrain Transformers to its 4.x API; installed versions and the resolved model commit are saved with every run. Dependencies are ranges, not a fully locked environment: preserve `pip freeze` for reproduction.

## What caching changes

The first forward pass processes the whole prompt (prefill). With caching ON, each subsequent forward pass receives only the latest token and the previous `past_key_values`; earlier keys and values are reused. With caching OFF, each pass recomputes the complete growing sequence. The full attention mask grows in both modes. Caches are fresh for every run; no cache is shared between prompts or repeats.

By default, inputs are synthetic repeated passage tokens, sliced directly to the requested token count. Custom prompts can instead use their natural length. Neither mode adds special tokens or a chat template. This measures inference scaling rather than response quality on natural conversations. Prompt hashes and input IDs identify the exact workload. EOS is deliberately ignored so both configurations always emit exactly the requested number of tokens. No sampling, beam search, temperature, or repetition penalties are used. Output IDs are checked against the warmup reference; any difference aborts the comparison. Greedy decoding and deterministic PyTorch algorithms reduce variability, but bitwise reproduction across hardware/library versions is not promised. SDPA can choose different kernels for different tensor shapes.

## Hardware and memory

Use an NVIDIA CUDA GPU and a compatible driver/CUDA-enabled PyTorch build. BF16 is the default; use FP16 on GPUs without BF16 support. There is no CPU fallback, quantization, offloading, multi-GPU placement, or compilation.

Planning estimates, **not benchmark results**: 7.61B parameters at two bytes each require roughly 14.2 GiB for weights alone. A 24 GiB GPU is a reasonable starting point for the default batch-one workload; 32–48 GiB offers more headroom. Runtime buffers and attention kernels vary, so capacity is not guaranteed. FP32 weights alone need roughly 28.4 GiB. Allow approximately 32 GiB or more host RAM and 30 GiB or more free disk for downloads/environment, with more RAM potentially needed during loading. Standard BF16 KV storage for this model is approximately 56 KiB per sequence token (28 layers × two K/V × four heads × 128 dimensions × two bytes), before allocator overhead.

An out-of-memory error names the GPU and suggests reducing lengths, using half precision, freeing other workloads, or selecting a larger GPU. Earlier measured CSV rows are flushed and retained. A partial CSV must not be treated as a completed experiment; the plotting script rejects missing or duplicate runs.

## Setup on an existing rented GPU (for example RunPod)

These are manual instructions for a GPU machine you already control. The scripts do not rent instances or make cloud API calls. Choose a Linux PyTorch/CUDA image with a compatible NVIDIA driver, upload this directory, open its terminal, and use a persistent volume for the Hugging Face download cache. Keep the GPU free of other workloads.

```bash
nvidia-smi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pip freeze > environment.txt
```

If the image lacks CUDA-enabled PyTorch, install the wheel selected by the [official PyTorch installer](https://pytorch.org/get-started/locally/) for its OS and supported CUDA version before the requirements. Installing a CPU build will not work. A compatible CUDA wheel does not normally require separately installing the CUDA compiler toolkit. For Windows, activate with `.\.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

The first actual benchmark downloads approximately 15 GB of weights from Hugging Face. Set `HF_HOME` to a persistent cache directory if desired. Download/model load are excluded from measurements.

## Run

Small smoke test (still loads the full 7B model):

```bash
python benchmark.py --prompt-lengths 128 --generated-tokens 4 --warmups 1 --repeats 1 --csv results/smoke.csv
```

Full default benchmark (40 measured rows):

```bash
python benchmark.py --prompt-lengths 128 512 1024 2048 --generated-tokens 64 --warmups 2 --repeats 5 --precision bfloat16 --csv results/full.csv
```

Use `--revision <Hub-commit-SHA>` to reproduce the revision recorded in an earlier CSV. `main` is resolved once; tokenizer and weights use that same commit. Options include `--device`, `--seed`, `--precision`, and `--model` (other models must support the Qwen-style forward arguments, notably `logits_to_keep`). `--device` uses the index after any `CUDA_VISIBLE_DEVICES` filtering. Existing CSVs are never overwritten. ON/OFF order alternates by measured pair to reduce systematic ordering effects; warmups are performed separately for each prompt length and cache setting.

```bash
python benchmark.py --help
python plot_results.py --help
python plot_results.py results/full.csv --output-dir results/plots
```

Plots are saved as `benchmark.png` (four clearly labelled ON/OFF panels) and grouped values as `medians.csv`. Only measured runs enter the medians. The one-token output case has no subsequent-token latency, represented by an empty CSV value/plot gap.

### Give your own prompt

Pass text, read a UTF-8 file (multiline supported), or enter a one-line prompt interactively:

```bash
python benchmark.py --prompt "Explain KV caching with an example." --csv results/custom.csv
python benchmark.py --prompt-file prompt.txt --csv results/from_file.csv
python benchmark.py --interactive --csv results/interactive.csv
```

These three inputs are mutually exclusive. Custom input defaults to **natural** mode: tokenize the complete prompt once and benchmark that exact token sequence, at one length. `--prompt-lengths` only applies in **exact** mode. The script prints the actual token lengths; the CSV and plots use those actual lengths. Empty/whitespace prompts and unreadable files produce useful errors. Prompts beyond the model context limit are rejected before loading weights.

To compare your text at all four standard lengths, explicitly select exact mode:

```bash
python benchmark.py --prompt-file prompt.txt --prompt-mode exact --prompt-lengths 128 512 1024 2048 --generated-tokens 64 --csv results/custom_lengths.csv
```

Exact mode repeats or truncates the tokenized text without re-tokenizing, so a truncated prompt may end mid-sentence. With no custom input, the original built-in passage and exact-length defaults still apply. Both configurations always reuse the same prepared input IDs. The CSV stores the source text and resulting token IDs for reproduction; keep that in mind before sharing CSVs containing private prompts. This remains a raw-token inference benchmark without an instruction chat template.

## Timing definitions and exclusions

Each token step uses `time.perf_counter()` with CUDA synchronized before the timer and before the final timestamp. First-token latency is prompt forward + final-position logits + greedy token selection. Subsequent-token latency is the arithmetic mean of the remaining steps, including sequence/mask updates, Python dispatch, forward computation, token selection, and the ending synchronization. Output throughput is generated tokens divided by the sum of token-step durations, **including prefill**. It is not decode-only throughput or end-to-end request throughput. Per-token synchronization adds overhead and limits pipelining, so compare these instrumented ON/OFF runs rather than production server claims.

Excluded: downloading, tokenizer/model loading, prompt tokenization, initial input GPU transfer, initial input clone/mask creation, warmups, allocator clearing/stat reset, leading synchronization, gaps between token timers, CPU output transfer, detokenization, output validation, printing, CSV I/O, and plotting. `logits_to_keep=1` projects only the last hidden position in both modes; OFF still recomputes all transformer positions. The allocator cache is emptied before each run outside timing, so warmups initialize kernels but measured runs do not inherit cached free allocator blocks.

Peak allocated memory is the PyTorch tensor allocation high-water mark during the run, including resident model weights and input tensors; it is **not** total device usage from `nvidia-smi`. Peak reserved memory additionally includes PyTorch's allocator reservations. The baseline is sampled before run-local buffers and KV tensors. Memory counters are reset per run. Other processes/driver allocations are not included in these PyTorch counters.

## CSV columns

| Columns | Meaning |
| --- | --- |
| `experiment_id`, `timestamp_utc` | Experiment UUID and measured-row completion time |
| `model`, `requested_revision`, `model_revision` | Hub ID, requested ref, resolved commit |
| `torch_version`, `transformers_version`, `python_version`, `cuda_version`, `package_versions` | Runtime/library metadata; package versions are JSON |
| `gpu`, `gpu_total_mib`, `device`, `precision` | Device identity, total capacity, visible index, weight/compute dtype |
| `attention_implementation`, `decoding`, `batch_size`, `seed`, `deterministic_algorithms`, `tf32` | Fixed inference settings |
| `warmups`, `repeats`, `generated_tokens`, `prompt_lengths` | Experiment settings; prompt lengths are JSON |
| `prompt_tokens`, `prompt_sha256` | Exact input count and hash of JSON token IDs |
| `prompt_source`, `prompt_mode`, `prompt_text`, `input_token_ids` | Built-in/text/file/interactive source, natural/exact mode, original text, resulting IDs as JSON |
| `repeat`, `order_in_pair`, `cache` | One-based measured repeat/order; `ON` or `OFF` |
| `output_token_ids`, `tokens_match` | Generated IDs as JSON and equality to the reference |
| `first_token_ms`, `subsequent_token_ms`, `total_seconds`, `output_tokens_per_second` | Instrumented timings and throughput |
| `baseline_allocated_mib`, `peak_allocated_mib`, `peak_reserved_mib` | Per-process PyTorch GPU memory, in MiB |

## Interpretation and verification

Expect caching to reduce repeated computation during decoding, especially at longer prompts. First-token work is similar in both modes; small timing differences need repeated measurements. Caching stores extra KV tensors, while OFF can need larger transient activation/attention buffers, so measured peaks depend on the workload and kernel. Inspect variability in the raw CSV alongside medians; medians alone are not statistical significance. Longer prompts can lower throughput because the metric includes prefill. Compare only experiments with identical settings and complete matching outputs.

No benchmark results are supplied. Local checks can run without downloading weights:

```bash
python -m py_compile benchmark.py plot_results.py test_benchmark.py
python -m unittest -v test_benchmark.py
```

The tests cover exact input lengths, metric math, CLI validation, and cached/full-sequence equivalence using a tiny randomly initialized Qwen2 model on CPU. This is a correctness check, not a Qwen2.5-7B performance measurement. Actual timings, GPU memory behavior, model download/loading, and CUDA kernel determinism require the GPU smoke test and full benchmark above.

### Verification on the development machine

Verified on 2026-10-08: all three Python source files compiled in memory; both CLI help commands succeeded; all five offline tests passed, including PNG/median generation from artificial fixture values and rejection of an incomplete CSV. The fixture is temporary and is not shipped as benchmark results. The installed Qwen2 forward signature was checked for `past_key_values`, `use_cache`, and `logits_to_keep` support. Test environment: Python 3.10.4, PyTorch 2.14.0+cpu, Transformers 4.57.6, pandas 2.3.3, matplotlib 3.10.9.

PyTorch reported `cuda.is_available() == False`, no CUDA build, and zero visible devices; `nvidia-smi` was unavailable. The smoke command was invoked and correctly exited with the CUDA-unavailable message before downloading or loading weights. No CUDA inference, 7B model loading, or benchmark performance/memory results were verified. Native bytecode output initially hit sandbox directory permissions; the successful syntax check used Python's in-memory `compile()` instead. The local `.venv` inherits the preinstalled CPU PyTorch and is for offline verification, not a GPU-ready environment.
