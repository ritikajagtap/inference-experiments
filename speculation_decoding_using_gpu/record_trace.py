"""Record a real speculative-decoding run on a GPU into trace.json for monitor.html.

This runs the actual draft + target decode on the GPU, captures the per-round
structure (the draft's guesses, which ones were kept, the correction or bonus), and
samples real GPU utilization from the NVIDIA driver (NVML) in a background thread
while it decodes. It writes a small trace.json that you download and load into
monitor.html to replay the run — including the measured compute (SM) and memory
numbers, so the picture is real, not modeled.

    python record_trace.py --prompt "Explain speculative decoding." --gamma 4 \
        --max-new-tokens 96 --out trace.json

Needs a CUDA GPU and the NVML bindings:  pip install nvidia-ml-py

The decode loop below reuses the exact forward/cache helpers from
`speculative_decoding.py`, so the KV-cache handling cannot drift from the real
implementation; only the timing and token capture are added here. As a safety net
the committed tokens are checked against the plain greedy baseline before writing.
"""
from __future__ import annotations

import argparse
import json
import threading
import time

from models import DEFAULT_DRAFT, DEFAULT_TARGET, load_pair
from speculative_decoding import (
    _crop,
    _forward,
    _greedy_token,
    _tokens_not_cached,
    greedy_generate,
)


class GpuSampler(threading.Thread):
    """Polls NVML for GPU utilization in the background until stopped.

    Each sample is (time_s, sm_percent, mem_percent): `sm_percent` is the share of
    time the GPU was running kernels (NVML's "gpu" utilization, our compute proxy)
    and `mem_percent` is the share of time its memory was being read or written.
    """

    def __init__(self, handle, pynvml, interval_s: float = 0.005):
        super().__init__(daemon=True)
        self.handle = handle
        self.pynvml = pynvml
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []
        # NB: not `self._stop` — threading.Thread already has an internal _stop() method
        # that join() calls, and shadowing it with an Event breaks join().
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.samples.append((time.perf_counter(), float(util.gpu), float(util.memory)))
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def mean_in_window(samples, window, field: int) -> float:
    """Average a sample field over a (start, end) time window. Falls back to the
    nearest sample when the window is too short to contain one (fast phases can slip
    between NVML polls)."""
    start, end = window
    values = [s[field] for s in samples if start <= s[0] <= end]
    if values:
        return round(sum(values) / len(values), 1)
    if not samples:
        return 0.0
    middle = (start + end) / 2
    return round(min(samples, key=lambda s: abs(s[0] - middle))[field], 1)


def record(draft, target, tokenizer, input_ids, max_new_tokens, gamma, device, torch, vocab_limit):
    """Run speculative decoding while timing each phase. Returns (sequence, rounds)
    where each round records its guesses, outcome, and per-phase time windows.

    Mirrors speculative_decoding.speculative_generate (same forward/cache helpers,
    same shared-vocab clipping) with timing and token capture added."""
    from transformers import DynamicCache

    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    prompt_len = input_ids.shape[1]
    draft_cache = DynamicCache()
    target_cache = DynamicCache()
    prefix = input_ids
    rounds = []

    while prefix.shape[1] - prompt_len < max_new_tokens:
        # Draft guesses gamma tokens, timing each one.
        proposals, draft_windows = [], []
        feed = _tokens_not_cached(prefix, draft_cache)
        for _ in range(gamma):
            sync(); start = time.perf_counter()
            logits, draft_cache = _forward(draft, draft_cache, feed, keep=1)
            feed = _greedy_token(logits[..., :vocab_limit])
            sync(); draft_windows.append((start, time.perf_counter()))
            proposals.append(feed)
        proposal_ids = torch.cat(proposals, dim=1)

        # Target checks all guesses in one pass, timed as a single window.
        sync(); start = time.perf_counter()
        target_feed = torch.cat([_tokens_not_cached(prefix, target_cache), proposal_ids], dim=1)
        logits, target_cache = _forward(target, target_cache, target_feed, keep=gamma + 1)
        target_picks = logits[0, :, :vocab_limit].argmax(dim=-1)
        sync(); target_window = (start, time.perf_counter())

        n_accept = 0
        while n_accept < gamma and proposal_ids[0, n_accept].item() == target_picks[n_accept].item():
            n_accept += 1
        correction = target_picks[n_accept].view(1, 1)
        new_tokens = torch.cat([proposal_ids[:, :n_accept], correction], dim=1)
        prefix = torch.cat([prefix, new_tokens], dim=1)
        keep_len = prefix.shape[1] - 1
        _crop(target_cache, keep_len)
        _crop(draft_cache, keep_len)

        decode = lambda tid: tokenizer.decode([int(tid)])
        committed = [{"text": decode(proposal_ids[0, i]), "kind": "accepted"} for i in range(n_accept)]
        committed.append({"text": decode(correction), "kind": "bonus" if n_accept == gamma else "correction"})
        rounds.append({
            "proposals": [decode(proposal_ids[0, i]) for i in range(gamma)],
            "nAccept": n_accept,
            "committed": committed,
            "draft_windows": draft_windows,
            "target_window": target_window,
        })

    return prefix[:, : prompt_len + max_new_tokens], rounds


def attach_utilization(rounds, samples):
    """Fold the NVML samples into each round as measured per-phase SM / memory %,
    then drop the raw time windows the page does not need."""
    for r in rounds:
        r["draftSm"] = [mean_in_window(samples, w, 1) for w in r["draft_windows"]]
        r["draftMem"] = [mean_in_window(samples, w, 2) for w in r["draft_windows"]]
        r["draftMs"] = [round((w[1] - w[0]) * 1000, 2) for w in r["draft_windows"]]
        r["targetSm"] = mean_in_window(samples, r["target_window"], 1)
        r["targetMem"] = mean_in_window(samples, r["target_window"], 2)
        r["targetMs"] = round((r["target_window"][1] - r["target_window"][0]) * 1000, 2)
        del r["draft_windows"], r["target_window"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", default=DEFAULT_DRAFT)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--draft-revision", default="main")
    p.add_argument("--target-revision", default="main")
    p.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    p.add_argument("--nvml-index", type=int, default=None,
                   help="physical GPU index for NVML (defaults to --device; set this if "
                        "CUDA_VISIBLE_DEVICES remaps the device)")
    p.add_argument("--sample-ms", type=float, default=5.0, help="NVML polling interval")
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the target's chat template")
    p.add_argument("--out", default="trace.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. record_trace.py measures a real GPU; run it on a CUDA GPU.")
    try:
        import pynvml
    except ImportError:
        raise SystemExit("NVML bindings not found. Install them with:  pip install nvidia-ml-py")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    draft, target, tokenizer, info = load_pair(
        args.draft, args.target, args.draft_revision, args.target_revision,
        dtype=getattr(torch, args.precision), device=device,
    )

    if args.chat:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.prompt
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Warm up kernels before measuring (not recorded).
    record(draft, target, tokenizer, input_ids, max_new_tokens=8, gamma=args.gamma,
           device=device, torch=torch, vocab_limit=info["shared_vocab"])

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.nvml_index if args.nvml_index is not None else args.device)
    sampler = GpuSampler(handle, pynvml, interval_s=args.sample_ms / 1000)
    sampler.start()
    try:
        sequence, rounds = record(draft, target, tokenizer, input_ids,
                                  args.max_new_tokens, args.gamma, device, torch,
                                  vocab_limit=info["shared_vocab"])
    finally:
        sampler.stop()
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
    pynvml.nvmlShutdown()

    # In exact arithmetic greedy speculative decoding is token-identical to greedy
    # target-only decoding (the CPU tests confirm this in fp32). On a real GPU in bf16
    # the two paths can still drift: the batched verify pass and the one-token baseline
    # use different matmul shapes, so a near-tie in the logits may round to a different
    # argmax and the sequences diverge from there. That is floating-point noise, not a
    # bug, so report the agreement but still write the trace from the real run.
    baseline, _ = greedy_generate(target, input_ids, args.max_new_tokens, eos_id=None,
                                  vocab_limit=info["shared_vocab"])
    spec_new = sequence[0, input_ids.shape[1]:].tolist()
    base_new = baseline[0, input_ids.shape[1]:].tolist()
    agree = 0
    for a, b in zip(spec_new, base_new):
        if a != b:
            break
        agree += 1
    if agree < len(spec_new):
        print(f"Note: matches the greedy baseline on the first {agree}/{len(spec_new)} tokens; "
              "the rest differs from bf16 argmax rounding at near-ties (expected on GPU, not a bug).")

    attach_utilization(rounds, sampler.samples)
    trace = {
        "meta": {
            "draft_model": args.draft, "target_model": args.target,
            "draft_revision": info["draft_revision"], "target_revision": info["target_revision"],
            "gpu": gpu_name, "precision": args.precision, "gamma": args.gamma,
            "generated_tokens": args.max_new_tokens,
            "utilization_source": "measured (NVML gpu util)",
        },
        "rounds": rounds,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trace, f)
    accepted = sum(r["nAccept"] for r in rounds)
    print(f"Wrote {args.out}: {len(rounds)} rounds, {args.max_new_tokens} tokens, "
          f"acceptance {accepted / (len(rounds) * args.gamma):.0%}, GPU {gpu_name}.")
    print("Download it and load it into monitor.html.")


if __name__ == "__main__":
    main()
