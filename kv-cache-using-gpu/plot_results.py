"""Plot median metrics from one complete benchmark experiment."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('results/plots'))
    args = p.parse_args()
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(args.csv)
    required = ['experiment_id', 'model', 'prompt_tokens', 'cache', 'repeat', 'repeats',
                'prompt_lengths', 'tokens_match', 'first_token_ms', 'subsequent_token_ms',
                'output_tokens_per_second', 'peak_allocated_mib']
    if any(c not in data for c in required):
        p.error('CSV is missing benchmark columns')
    if data.empty or data.experiment_id.nunique() != 1:
        p.error('CSV must contain exactly one nonempty experiment')
    if not data.tokens_match.astype(str).str.lower().eq('true').all():
        p.error('CSV contains mismatching output tokens')
    import json
    lengths = set(json.loads(data.prompt_lengths.iloc[0]))
    expected = {(n, c, r) for n in lengths for c in ['ON', 'OFF']
                for r in range(1, int(data.repeats.iloc[0]) + 1)}
    actual = list(zip(data.prompt_tokens, data.cache, data['repeat']))
    if len(actual) != len(expected) or set(actual) != expected:
        p.error('CSV is incomplete or contains duplicate runs; finish a new experiment before plotting')
    columns = [('first_token_ms', 'First-token latency (ms)'),
               ('subsequent_token_ms', 'Mean subsequent-token latency (ms)'),
               ('output_tokens_per_second', 'Output tokens/s (including prefill)'),
               ('peak_allocated_mib', 'Peak allocated GPU memory (MiB)')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    medians = data.groupby(['prompt_tokens', 'cache'])[[x[0] for x in columns]].median()
    for ax, (column, label) in zip(axes.flat, columns):
        for cache, color in [('ON', 'tab:blue'), ('OFF', 'tab:orange')]:
            series = medians.xs(cache, level='cache')[column].sort_index()
            ax.plot(series.index, series.values, marker='o', color=color, label=f'Cache {cache}')
        ax.set(xlabel='Prompt length (tokens)', ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'{data.model.iloc[0]} — medians of measured runs')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'benchmark.png', dpi=180)
    plt.close(fig)
    medians.to_csv(args.output_dir / 'medians.csv')
    print(f'Saved plots and medians to {args.output_dir}')


if __name__ == '__main__':
    main()
