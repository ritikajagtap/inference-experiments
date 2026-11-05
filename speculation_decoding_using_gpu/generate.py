"""Sample from a draft/target pair with speculative decoding, and compare to the
greedy target-only baseline.

    python generate.py --prompt "Explain speculative decoding in one sentence." --max-new-tokens 128
    python generate.py --prompt "..." --gamma 6 --no-baseline

Greedy speculative output is token-identical to the baseline (see
speculative_decoding.py); running both here is a sanity check plus a quick look
at the acceptance rate for your prompt and chosen gamma.
"""
from __future__ import annotations

import argparse
import time

from models import DEFAULT_DRAFT, DEFAULT_TARGET, load_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", default=DEFAULT_DRAFT)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--draft-revision", default="main")
    p.add_argument("--target-revision", default="main")
    p.add_argument("--prompt", default="Explain speculative decoding in one sentence.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gamma", type=int, default=4, help="draft tokens proposed per round")
    p.add_argument("--precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the target's chat template")
    p.add_argument("--no-baseline", action="store_true", help="skip the greedy target-only comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Speculative decoding here targets a CUDA GPU; "
            "install CUDA-enabled PyTorch and run on a GPU."
        )
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    from speculative_decoding import greedy_generate, speculative_generate

    draft, target, tokenizer, info = load_pair(
        args.draft, args.target, args.draft_revision, args.target_revision,
        dtype=getattr(torch, args.precision), device=device,
    )
    eos_id = tokenizer.eos_token_id

    if args.chat:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.prompt
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    spec_ids, stats = speculative_generate(draft, target, input_ids, args.max_new_tokens, args.gamma, eos_id)
    torch.cuda.synchronize(device)
    spec_time = time.perf_counter() - start

    new = spec_ids.shape[1] - input_ids.shape[1]
    print("=== speculative ===")
    print(tokenizer.decode(spec_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
    print(
        f"\n{new} tokens in {spec_time:.2f}s = {new / spec_time:.1f} tok/s | "
        f"gamma={args.gamma} acceptance={stats.acceptance_rate:.0%} "
        f"({stats.mean_committed_per_round:.2f} tokens/target-forward over {stats.rounds} rounds)"
    )

    if not args.no_baseline:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        base_ids, base_stats = greedy_generate(target, input_ids, args.max_new_tokens, eos_id,
                                               vocab_limit=info["shared_vocab"])
        torch.cuda.synchronize(device)
        base_time = time.perf_counter() - start

        base_new = base_ids.shape[1] - input_ids.shape[1]
        print("\n=== baseline (greedy target-only) ===")
        print(
            f"{base_new} tokens in {base_time:.2f}s = {base_new / base_time:.1f} tok/s | "
            f"{base_stats.target_forwards} target forwards"
        )
        matched = torch.equal(spec_ids[:, : base_ids.shape[1]], base_ids[:, : spec_ids.shape[1]])
        print(f"\nspeculative output matches baseline: {matched} | speedup: {base_time / spec_time:.2f}x")


if __name__ == "__main__":
    main()
