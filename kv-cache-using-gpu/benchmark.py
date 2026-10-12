"""Batch-one greedy inference benchmark on a single CUDA GPU."""
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

DEFAULT_PROMPT = 'Explain how computers store information and process language. '


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--revision', default='main', help='Hub revision; resolved commit is recorded')
    p.add_argument('--prompt-lengths', nargs='+', type=positive, default=[128, 512, 1024, 2048])
    source = p.add_mutually_exclusive_group()
    source.add_argument('--prompt', help='custom prompt text (natural token length by default)')
    source.add_argument('--prompt-file', type=Path, help='UTF-8 text file containing a custom prompt')
    source.add_argument('--interactive', action='store_true', help='ask for a one-line prompt before loading')
    p.add_argument('--prompt-mode', choices=['natural', 'exact'],
                   help='natural: use all prompt IDs; exact: repeat/truncate to --prompt-lengths. '
                        'Default: natural for custom input, exact for the built-in passage')
    p.add_argument('--generated-tokens', type=positive, default=64)
    p.add_argument('--repeats', type=positive, default=5)
    p.add_argument('--warmups', type=positive, default=2)
    p.add_argument('--precision', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--device', type=int, default=0, help='visible CUDA device index')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--csv', type=Path, default=Path('results/runs.csv'))
    return p


def read_prompt(args):
    if args.prompt_file is not None:
        text, source = args.prompt_file.read_text(encoding='utf-8-sig'), 'file'
    elif args.interactive:
        text, source = input('Enter prompt: '), 'interactive'
    elif args.prompt is not None:
        text, source = args.prompt, 'text'
    else:
        text, source = DEFAULT_PROMPT, 'built-in'
    if not text.strip():
        raise ValueError('Prompt must contain non-whitespace text')
    return text, source


def prompt_ids(tokenizer, length=None, text=DEFAULT_PROMPT):
    """Tokenize once, optionally repeat/slice IDs without re-tokenizing."""
    base = tokenizer.encode(text, add_special_tokens=False)
    if not base:
        raise ValueError('Tokenizer returned no tokens')
    if length is None:
        return base
    return (base * ((length + len(base) - 1) // len(base)))[:length]


def prepare_prompts(tokenizer, text, mode, lengths):
    base = prompt_ids(tokenizer, text=text)
    if mode == 'natural':
        return [(len(base), base)]
    return [(length, (base * ((length + len(base) - 1) // len(base)))[:length])
            for length in dict.fromkeys(lengths)]


def metrics(times):
    total = sum(times)
    return dict(first_token_ms=times[0] * 1000,
                subsequent_token_ms=sum(times[1:]) / (len(times) - 1) * 1000 if len(times) > 1 else '',
                total_seconds=total, output_tokens_per_second=len(times) / total)


def greedy_step(model, sequence, mask, next_token, past, cache_on, step, torch):
    if step:
        sequence = torch.cat((sequence, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
    current = next_token if cache_on and step else sequence
    out = model(input_ids=current, attention_mask=mask,
                past_key_values=past if cache_on else None,
                use_cache=cache_on, logits_to_keep=1)
    token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return sequence, mask, token, out.past_key_values if cache_on else None


def decode(model, inputs, count, cache_on, torch):
    device = inputs.device
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    sequence = inputs.clone()
    mask = torch.ones_like(sequence)
    past = next_token = None
    times, emitted = [], []
    with torch.inference_mode():
        for step in range(count):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            sequence, mask, next_token, past = greedy_step(
                model, sequence, mask, next_token, past, cache_on, step, torch)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
            emitted.append(next_token)
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    tokens = torch.cat(emitted, dim=1).cpu().tolist()[0]
    return {**metrics(times), 'baseline_allocated_mib': baseline / 2**20,
            'peak_allocated_mib': peak / 2**20, 'peak_reserved_mib': reserved / 2**20}, tokens


def main():
    p = make_parser()
    args = p.parse_args()
    if args.device < 0:
        p.error('--device must be nonnegative')
    if args.csv.exists():
        p.error(f'{args.csv} already exists; choose a new --csv to preserve previous runs')
    try:
        text, source = read_prompt(args)
    except (OSError, UnicodeError, ValueError, EOFError) as exc:
        p.error(f'Cannot read prompt: {exc}')
    mode = args.prompt_mode or ('exact' if source == 'built-in' else 'natural')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Install CUDA-enabled PyTorch and use a CUDA GPU; CPU fallback is disabled.')
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if args.device >= torch.cuda.device_count():
        p.error('Requested CUDA device is not visible')
    torch.cuda.set_device(args.device)
    if args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        p.error('GPU lacks bfloat16 support; use --precision float16 for both configurations')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda', args.device)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    revision = config._commit_hash
    if not revision:
        p.error('Unable to resolve Hub commit; use a Hub model and revision')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    try:
        prompts = prepare_prompts(tokenizer, text, mode, args.prompt_lengths)
    except ValueError as exc:
        p.error(str(exc))
    lengths = [length for length, _ in prompts]
    if max(lengths) + args.generated_tokens > config.max_position_embeddings:
        p.error('Prompt plus output exceeds model configured context length; shorten the prompt or use exact mode')
    print(f'Prompt source={source}, mode={mode}, actual token lengths={lengths}', flush=True)
    props = torch.cuda.get_device_properties(device)
    metadata = dict(experiment_id=str(uuid.uuid4()), model=args.model, requested_revision=args.revision,
                    model_revision=revision, torch_version=torch.__version__, transformers_version=transformers.__version__,
                    python_version=platform.python_version(), cuda_version=torch.version.cuda,
                    package_versions=json.dumps({n: importlib.metadata.version(n) for n in
                        ['torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors']}),
                    gpu=props.name, gpu_total_mib=props.total_memory / 2**20, device=args.device,
                    precision=args.precision, attention_implementation='sdpa', decoding='greedy_argmax_ignore_eos',
                    batch_size=1, seed=args.seed, warmups=args.warmups, repeats=args.repeats,
                    generated_tokens=args.generated_tokens, prompt_lengths=json.dumps(lengths),
                    prompt_source=source, prompt_mode=mode, prompt_text=text,
                    deterministic_algorithms=True, tf32=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision,
                    dtype=getattr(torch, args.precision), attn_implementation='sdpa').to(device).eval()
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('x', newline='', encoding='utf-8') as f:
            writer = None
            for length, ids in prompts:
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                expected = None
                for cache in [True, False]:
                    for _ in range(args.warmups):
                        _, tokens = decode(model, inputs, args.generated_tokens, cache, torch)
                        if expected is None:
                            expected = tokens
                        elif tokens != expected:
                            raise RuntimeError('ON/OFF greedy tokens differ; comparison aborted.')
                for repeat in range(args.repeats):
                    for order, cache in enumerate([True, False] if repeat % 2 == 0 else [False, True]):
                        result, tokens = decode(model, inputs, args.generated_tokens, cache, torch)
                        row = {**metadata, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               'prompt_tokens': length, 'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                               'input_token_ids': json.dumps(ids),
                               'repeat': repeat + 1, 'order_in_pair': order + 1, 'cache': 'ON' if cache else 'OFF',
                               'output_token_ids': json.dumps(tokens), 'tokens_match': tokens == expected, **result}
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        if tokens != expected:
                            raise RuntimeError('Greedy output changed; mismatching run saved, comparison aborted.')
                        print(f'prompt={length} repeat={repeat + 1} cache={row["cache"]}: {result}', flush=True)
                del inputs
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(f'GPU out of memory on {props.name} ({props.total_memory / 2**30:.1f} GiB). '
                         'Use a larger GPU, stop other GPU workloads, reduce prompt/output lengths, or use float16/bfloat16. '
                         f'Completed measured rows, if any, remain in {args.csv}. Details: {exc}') from exc


if __name__ == '__main__':
    main()
