"""Plot median metrics from one complete speculative-decoding experiment.

    python plot_results.py results/full.csv --output-dir results/plots

Produces `benchmark.png` (throughput, speedup, acceptance, memory) and the
grouped `medians.csv`. Only complete experiments with matching greedy outputs
are plotted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(args.csv)
    required = ["experiment_id", "target_model", "method", "gamma", "prompt_tokens", "repeat",
                "repeats", "prompt_lengths", "gammas", "tokens_match",
                "output_tokens_per_second", "acceptance_rate", "mean_committed_per_round",
                "peak_allocated_mib"]
    if any(c not in data for c in required):
        p.error("CSV is missing benchmark columns")
    if data.empty or data.experiment_id.nunique() != 1:
        p.error("CSV must contain exactly one nonempty experiment")
    if not data.tokens_match.astype(str).str.lower().eq("true").all():
        p.error("CSV contains mismatching output tokens")

    lengths = sorted(json.loads(data.prompt_lengths.iloc[0]))
    gammas = sorted(json.loads(data.gammas.iloc[0]))
    repeats = int(data.repeats.iloc[0])
    # Expected rows: one baseline + one row per gamma, per length, per repeat.
    expected = {(n, "baseline", "", r) for n in lengths for r in range(1, repeats + 1)}
    expected |= {(n, "speculative", g, r) for n in lengths for g in gammas for r in range(1, repeats + 1)}
    gamma_key = data.gamma.apply(lambda v: "" if pd.isna(v) or v == "" else int(float(v)))
    actual = list(zip(data.prompt_tokens, data.method, gamma_key, data["repeat"]))
    if len(actual) != len(expected) or set(actual) != expected:
        p.error("CSV is incomplete or contains duplicate runs; finish a new experiment before plotting")

    data = data.assign(gamma_key=gamma_key)
    base = data[data.method == "baseline"].groupby("prompt_tokens").output_tokens_per_second.median()
    spec = data[data.method == "speculative"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.cm.viridis([i / max(1, len(gammas) - 1) for i in range(len(gammas))])

    # (0,0) throughput vs prompt length — baseline and each gamma.
    ax = axes[0, 0]
    ax.plot(base.index, base.values, marker="s", color="black", label="baseline")
    for g, c in zip(gammas, colors):
        s = spec[spec.gamma_key == g].groupby("prompt_tokens").output_tokens_per_second.median().sort_index()
        ax.plot(s.index, s.values, marker="o", color=c, label=f"spec γ={g}")
    ax.set(xlabel="Prompt length (tokens)", ylabel="Output tokens/s (incl. prefill)")
    ax.grid(alpha=0.3); ax.legend()

    # (0,1) speedup over baseline vs gamma — one line per prompt length.
    ax = axes[0, 1]
    for n in lengths:
        ys = [spec[(spec.gamma_key == g) & (spec.prompt_tokens == n)].output_tokens_per_second.median() / base[n]
              for g in gammas]
        ax.plot(gammas, ys, marker="o", label=f"{n} tok")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set(xlabel="gamma (draft tokens/round)", ylabel="Speedup vs baseline (×)")
    ax.grid(alpha=0.3); ax.legend()

    # (1,0) acceptance rate vs gamma — one line per prompt length.
    ax = axes[1, 0]
    for n in lengths:
        ys = [spec[(spec.gamma_key == g) & (spec.prompt_tokens == n)].acceptance_rate.median() for g in gammas]
        ax.plot(gammas, ys, marker="o", label=f"{n} tok")
    ax.set(xlabel="gamma (draft tokens/round)", ylabel="Acceptance rate", ylim=(0, 1))
    ax.grid(alpha=0.3); ax.legend()

    # (1,1) peak allocated memory vs prompt length — baseline and each gamma.
    ax = axes[1, 1]
    bm = data[data.method == "baseline"].groupby("prompt_tokens").peak_allocated_mib.median()
    ax.plot(bm.index, bm.values, marker="s", color="black", label="baseline")
    for g, c in zip(gammas, colors):
        s = spec[spec.gamma_key == g].groupby("prompt_tokens").peak_allocated_mib.median().sort_index()
        ax.plot(s.index, s.values, marker="o", color=c, label=f"spec γ={g}")
    ax.set(xlabel="Prompt length (tokens)", ylabel="Peak allocated GPU memory (MiB)")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f"{data.target_model.iloc[0]} drafted by {data.draft_model.iloc[0]} — medians")
    fig.tight_layout()
    fig.savefig(args.output_dir / "benchmark.png", dpi=180)
    plt.close(fig)

    medians = data.groupby(["prompt_tokens", "method", "gamma_key"])[
        ["output_tokens_per_second", "acceptance_rate", "mean_committed_per_round", "peak_allocated_mib"]
    ].median()
    medians.to_csv(args.output_dir / "medians.csv")
    print(f"Saved plots and medians to {args.output_dir}")


if __name__ == "__main__":
    main()
