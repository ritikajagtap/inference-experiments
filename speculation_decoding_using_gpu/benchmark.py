"""Batch-one speculative-decoding benchmark on a single CUDA GPU.

Compares greedy **speculative decoding** (small draft + large target) against the
greedy **target-only baseline**, at one or more prompt lengths and draft lengths
(`gamma`). Both paths decode greedily, so the speculative output must match the
baseline token-for-token; any mismatch aborts the comparison. The main metric is
end-to-end output tokens per second (counting the first pass over the prompt for
both models); the CSV also records the acceptance rate and how many tokens each
target pass produced, which is where the speedup comes from.

    # smoke test (still downloads both models)
    python benchmark.py --prompt-lengths 128 --generated-tokens 8 --gammas 4 \
        --warmups 1 --repeats 1 --csv results/smoke.csv

    # fuller run: two prompt lengths, a gamma sweep
    python benchmark.py --prompt-lengths 512 2048 --generated-tokens 128 \
        --gammas 2 4 6 8 --warmups 2 --repeats 5 --csv results/full.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
from pathlib import Path

from models import DEFAULT_DRAFT, DEFAULT_TARGET

DEFAULT_PROMPT = "Explain how computers store information and process language. "


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", default=DEFAULT_DRAFT)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--draft-revision", default="main", help="Hub revision; resolved commit is recorded")
    p.add_argument("--target-revision", default="main", help="Hub revision; resolved commit is recorded")
    p.add_argument("--prompt-lengths", nargs="+", type=positive, default=[128, 512, 1024, 2048])
    p.add_argument("--gammas", nargs="+", type=positive, default=[4],
                   help="draft tokens proposed per round; one benchmark per value")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="custom prompt text (natural token length by default)")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 text file containing a custom prompt")
    source.add_argument("--interactive", action="store_true", help="ask for a one-line prompt before loading")
    p.add_argument("--prompt-mode", choices=["natural", "exact"],
                   help="natural: use all prompt IDs; exact: repeat/truncate to --prompt-lengths. "
                        "Default: natural for custom input, exact for the built-in passage")
    p.add_argument("--generated-tokens", type=positive, default=64)
    p.add_argument("--repeats", type=positive, default=5)
    p.add_argument("--warmups", type=positive, default=2)
    p.add_argument("--precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=Path, default=Path("results/runs.csv"))
    return p


def read_prompt(args):
    if args.prompt_file is not None:
        text, source = args.prompt_file.read_text(encoding="utf-8-sig"), "file"
    elif args.interactive:
        text, source = input("Enter prompt: "), "interactive"
    elif args.prompt is not None:
        text, source = args.prompt, "text"
    else:
        text, source = DEFAULT_PROMPT, "built-in"
    if not text.strip():
        raise ValueError("Prompt must contain non-whitespace text")
    return text, source


def prompt_ids(tokenizer, length=None, text=DEFAULT_PROMPT):
    """Tokenize once, optionally repeat/slice IDs without re-tokenizing."""
    base = tokenizer.encode(text, add_special_tokens=False)
    if not base:
        raise ValueError("Tokenizer returned no tokens")
    if length is None:
        return base
    return (base * ((length + len(base) - 1) // len(base)))[:length]


def prepare_prompts(tokenizer, text, mode, lengths):
    base = prompt_ids(tokenizer, text=text)
    if mode == "natural":
        return [(len(base), base)]
    return [(length, (base * ((length + len(base) - 1) // len(base)))[:length])
            for length in dict.fromkeys(lengths)]


def memory(torch, device, baseline):
    return dict(baseline_allocated_mib=baseline / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)


def run(fn, prompt_len, torch, device):
    """Time one full generation, returning (new_token_ids, stats, metrics)."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    sequence, stats = fn()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    new_ids = sequence[0, prompt_len:].cpu().tolist()
    result = dict(total_seconds=elapsed,
                  output_tokens_per_second=len(new_ids) / elapsed if elapsed else "",
                  **memory(torch, device, baseline))
    return new_ids, stats, result


def stat_fields(stats, gamma):
    """Per-method accounting columns; baseline leaves the gamma-only ones blank."""
    return dict(gamma=gamma if gamma is not None else "",
                rounds=stats.rounds, target_forwards=stats.target_forwards,
                draft_forwards=stats.draft_forwards, committed=stats.committed,
                acceptance_rate=stats.acceptance_rate if gamma is not None else "",
                mean_accepted_per_round=stats.mean_accepted_per_round if gamma is not None else "",
                mean_committed_per_round=stats.mean_committed_per_round)


def main():
    p = make_parser()
    args = p.parse_args()
    if args.device < 0:
        p.error("--device must be nonnegative")
    if args.csv.exists():
        p.error(f"{args.csv} already exists; choose a new --csv to preserve previous runs")
    try:
        text, source = read_prompt(args)
    except (OSError, UnicodeError, ValueError, EOFError) as exc:
        p.error(f"Cannot read prompt: {exc}")
    mode = args.prompt_mode or ("exact" if source == "built-in" else "natural")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install CUDA-enabled PyTorch and use a CUDA GPU; CPU fallback is disabled.")
    import transformers

    from models import load_pair
    from speculative_decoding import greedy_generate, speculative_generate

    if args.device >= torch.cuda.device_count():
        p.error("Requested CUDA device is not visible")
    torch.cuda.set_device(args.device)
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        p.error("GPU lacks bfloat16 support; use --precision float16")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", args.device)

    draft, target, tokenizer, info = load_pair(
        args.draft, args.target, args.draft_revision, args.target_revision,
        dtype=getattr(torch, args.precision), device=device,
    )
    try:
        prompts = prepare_prompts(tokenizer, text, mode, args.prompt_lengths)
    except ValueError as exc:
        p.error(str(exc))
    lengths = [length for length, _ in prompts]
    headroom = max(lengths) + args.generated_tokens + max(args.gammas)
    if headroom > info["target_max_position_embeddings"]:
        p.error("Prompt + output + gamma exceeds the target's configured context length; shorten the prompt")
    print(f"Prompt source={source}, mode={mode}, actual token lengths={lengths}", flush=True)

    props = torch.cuda.get_device_properties(device)
    metadata = dict(
        experiment_id=str(uuid.uuid4()), draft_model=args.draft, target_model=args.target,
        requested_draft_revision=args.draft_revision, requested_target_revision=args.target_revision,
        draft_revision=info["draft_revision"], target_revision=info["target_revision"],
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        python_version=platform.python_version(), cuda_version=torch.version.cuda,
        package_versions=json.dumps({n: importlib.metadata.version(n) for n in
            ["torch", "transformers", "huggingface-hub", "tokenizers", "safetensors"]}),
        gpu=props.name, gpu_total_mib=props.total_memory / 2**20, device=args.device,
        precision=args.precision, attention_implementation="sdpa", decoding="greedy_argmax_ignore_eos",
        batch_size=1, seed=args.seed, warmups=args.warmups, repeats=args.repeats,
        generated_tokens=args.generated_tokens, prompt_lengths=json.dumps(lengths),
        gammas=json.dumps(args.gammas), prompt_source=source, prompt_mode=mode, prompt_text=text,
        deterministic_algorithms=True, tf32=False,
    )

    # One target forward per token, EOS ignored, so every path emits exactly
    # generated_tokens: baseline and every gamma produce the same reference text.
    def baseline_fn(inputs):
        return lambda: greedy_generate(target, inputs, args.generated_tokens, eos_id=None,
                                       vocab_limit=info["shared_vocab"])

    def spec_fn(inputs, gamma):
        return lambda: speculative_generate(draft, target, inputs, args.generated_tokens, gamma, eos_id=None)

    try:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("x", newline="", encoding="utf-8") as f:
            writer = None
            for length, ids in prompts:
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                prompt_meta = dict(prompt_tokens=length,
                                   prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                   input_token_ids=json.dumps(ids))
                # Methods for this length: baseline (reference), then each gamma.
                methods = [("baseline", None, baseline_fn(inputs))]
                methods += [("speculative", g, spec_fn(inputs, g)) for g in args.gammas]

                reference = None
                for _ in range(args.warmups):
                    for method, gamma, fn in methods:
                        new_ids, _, _ = run(fn, length, torch, device)
                        if reference is None:
                            reference = new_ids
                        elif new_ids != reference:
                            raise RuntimeError(f"{method} (gamma={gamma}) greedy tokens differ from baseline; aborting.")

                for repeat in range(args.repeats):
                    for method, gamma, fn in methods:
                        new_ids, stats, result = run(fn, length, torch, device)
                        matches = new_ids == reference
                        row = {**metadata, **prompt_meta,
                               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "method": method, "repeat": repeat + 1,
                               "output_token_ids": json.dumps(new_ids), "tokens_match": matches,
                               **stat_fields(stats, gamma), **result}
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        if not matches:
                            raise RuntimeError(f"{method} (gamma={gamma}) output differs from baseline; mismatch saved, aborting.")
                        tps = result["output_tokens_per_second"]
                        extra = f" acceptance={stats.acceptance_rate:.0%}" if gamma is not None else ""
                        print(f"prompt={length} repeat={repeat + 1} {method}"
                              f"{'' if gamma is None else f' gamma={gamma}'}: {tps:.1f} tok/s{extra}", flush=True)
                del inputs
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            f"GPU out of memory on {props.name} ({props.total_memory / 2**30:.1f} GiB). "
            "Use a larger GPU, stop other GPU workloads, reduce prompt/output lengths or gamma, or use float16. "
            f"Completed measured rows, if any, remain in {args.csv}. Details: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
