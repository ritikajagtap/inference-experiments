"""Offline correctness tests; no pretrained model or GPU downloads."""
import unittest
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from benchmark import make_parser, prompt_ids, prepare_prompts, read_prompt, metrics, greedy_step


class BenchmarkTests(unittest.TestCase):
    def test_custom_prompt_modes(self):
        class Tokenizer:
            calls = 0
            def encode(self, text, add_special_tokens):
                self.calls += 1
                self.text = text
                return [4, 8, 12]
        tokenizer = Tokenizer()
        self.assertEqual(prepare_prompts(tokenizer, 'Custom text', 'natural', [128]),
                         [(3, [4, 8, 12])])
        self.assertEqual(tokenizer.text, 'Custom text')
        tokenizer.calls = 0
        self.assertEqual(prepare_prompts(tokenizer, 'Custom text', 'exact', [2, 5, 2]),
                         [(2, [4, 8]), (5, [4, 8, 12, 4, 8])])
        self.assertEqual(tokenizer.calls, 1)

    def test_prompt_sources_and_validation(self):
        from unittest.mock import patch
        parser = make_parser()
        self.assertEqual(read_prompt(parser.parse_args(['--prompt', 'Hello'])), ('Hello', 'text'))
        with self.assertRaises(ValueError):
            read_prompt(parser.parse_args(['--prompt', '   ']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prompt.txt'
            path.write_text('First line\nSecond line: café', encoding='utf-8-sig')
            self.assertEqual(read_prompt(parser.parse_args(['--prompt-file', str(path)])),
                             ('First line\nSecond line: café', 'file'))
            with self.assertRaises(FileNotFoundError):
                read_prompt(parser.parse_args(['--prompt-file', str(path.with_name('missing.txt'))]))
        with patch('builtins.input', return_value='Interactive text'):
            self.assertEqual(read_prompt(parser.parse_args(['--interactive'])),
                             ('Interactive text', 'interactive'))
        with self.assertRaises(SystemExit):
            parser.parse_args(['--prompt', 'Hello', '--interactive'])

    def test_plot_fixture_and_incomplete_rejection(self):
        # Artificial values exercise plotting; they are not inference results.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'synthetic_fixture.csv'
            rows = []
            for cache in ['ON', 'OFF']:
                for repeat, value in [(1, 1), (2, 3)]:
                    rows.append(dict(experiment_id='synthetic-test', model='test-fixture',
                        prompt_tokens=128, cache=cache, repeat=repeat, repeats=2,
                        prompt_lengths='[128]', tokens_match=True, first_token_ms=value,
                        subsequent_token_ms=value, output_tokens_per_second=value,
                        peak_allocated_mib=value))
            def save(items):
                with source.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(items)
            save(rows)
            command = [sys.executable, str(Path(__file__).with_name('plot_results.py')),
                       str(source), '--output-dir', str(root / 'plots')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((root / 'plots' / 'benchmark.png').stat().st_size, 1000)
            with (root / 'plots' / 'medians.csv').open() as f:
                for row in csv.DictReader(f):
                    self.assertEqual(float(row['first_token_ms']), 2)
            save(rows[:-1])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('incomplete', result.stderr)

    def test_exact_inputs(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                return [3, 7, 2]
        for length in [1, 128, 512, 1024, 2048]:
            ids = prompt_ids(Tokenizer(), length)
            self.assertEqual(len(ids), length)
            self.assertEqual(ids, prompt_ids(Tokenizer(), length))

    def test_metrics(self):
        result = metrics([1, 2, 3])
        self.assertEqual(result['first_token_ms'], 1000)
        self.assertEqual(result['subsequent_token_ms'], 2500)
        self.assertEqual(result['output_tokens_per_second'], 0.5)
        self.assertEqual(metrics([1])['subsequent_token_ms'], '')

    def test_defaults(self):
        args = make_parser().parse_args([])
        self.assertEqual(args.prompt_lengths, [128, 512, 1024, 2048])
        self.assertEqual(args.generated_tokens, 64)
        with self.assertRaises(SystemExit):
            make_parser().parse_args(['--generated-tokens', '0'])

    def test_qwen_cache_equivalence(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128)
        config._attn_implementation = 'sdpa'
        model = Qwen2ForCausalLM(config).eval()
        outputs = []
        with torch.inference_mode():
            for cache in [True, False]:
                sequence = torch.tensor([[1, 2, 3, 4]])
                mask = torch.ones_like(sequence)
                past = token = None
                emitted = []
                for step in range(6):
                    sequence, mask, token, past = greedy_step(
                        model, sequence, mask, token, past, cache, step, torch)
                    self.assertEqual(sequence.shape[1], 4 + step)
                    if cache:
                        self.assertEqual(past.get_seq_length(), 4 + step)
                    else:
                        self.assertIsNone(past)
                    emitted.append(token.item())
                outputs.append(emitted)
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
