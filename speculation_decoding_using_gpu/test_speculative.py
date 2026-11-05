"""Offline correctness tests; no pretrained models or GPU downloads.

The headline test is the greedy equivalence guarantee: speculative decoding with
a draft that differs from the target must still produce exactly the baseline's
greedy tokens. Everything else checks the accounting, the gamma=0 guard, and the
benchmark/plot CLIs, using tiny randomly initialized Qwen2 models on CPU.
"""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark import make_parser, prepare_prompts, prompt_ids, read_prompt
from speculative_decoding import SpecStats, greedy_generate, speculative_generate


def tiny_model(seed, vocab=64):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval()


class SpeculativeDecodingTests(unittest.TestCase):
    def test_greedy_output_matches_baseline(self):
        """Different draft and target: speculative output is token-identical to the
        greedy target-only baseline, and the draft is sometimes rejected (so the
        accept/reject and correction paths are actually exercised)."""
        import torch
        draft, target = tiny_model(1), tiny_model(2)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        for gamma in (1, 4, 7):
            spec, stats = speculative_generate(draft, target, input_ids, max_new_tokens=24, gamma=gamma)
            base, _ = greedy_generate(target, input_ids, max_new_tokens=24)
            self.assertTrue(torch.equal(spec, base), f"gamma={gamma}: {spec.tolist()} != {base.tolist()}")
            self.assertEqual(spec.shape[1] - input_ids.shape[1], 24)
            self.assertEqual(stats.proposed, stats.rounds * gamma)
            self.assertLessEqual(stats.accepted, stats.proposed)
            self.assertGreaterEqual(stats.committed, 24)
        # A genuine mismatch somewhere confirms rejection/correction ran, not just
        # a degenerate always-accept case.
        spec4, stats4 = speculative_generate(draft, target, input_ids, max_new_tokens=40, gamma=4)
        self.assertLess(stats4.accepted, stats4.proposed)

    def test_self_draft_reduces_target_forwards(self):
        """When the draft IS the target, proposals are the target's own greedy tokens,
        so acceptance is high and far fewer serial target passes are needed than the
        one-per-token baseline — the entire point of speculation."""
        import torch
        model = tiny_model(3)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        spec, stats = speculative_generate(model, model, input_ids, max_new_tokens=30, gamma=4)
        base, base_stats = greedy_generate(model, input_ids, max_new_tokens=30)
        self.assertTrue(torch.equal(spec, base))
        self.assertLess(stats.target_forwards, base_stats.target_forwards)
        self.assertGreater(stats.mean_committed_per_round, 1.5)

    def test_gamma_must_be_positive(self):
        import torch
        model = tiny_model(4)
        with self.assertRaises(ValueError):
            speculative_generate(model, model, torch.tensor([[1, 2]]), max_new_tokens=4, gamma=0)

    def test_stats_properties(self):
        stats = SpecStats(proposed=12, accepted=9, committed=15, rounds=3, target_forwards=3, draft_forwards=12)
        self.assertEqual(stats.acceptance_rate, 0.75)
        self.assertEqual(stats.mean_accepted_per_round, 3.0)
        self.assertEqual(stats.mean_committed_per_round, 5.0)
        empty = SpecStats()
        self.assertEqual(empty.acceptance_rate, 0.0)
        self.assertEqual(empty.mean_committed_per_round, 0.0)

    def test_prompt_modes_and_sources(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.text = text
                return [4, 8, 12]
        tokenizer = Tokenizer()
        self.assertEqual(prepare_prompts(tokenizer, "Custom", "natural", [128]), [(3, [4, 8, 12])])
        self.assertEqual(prepare_prompts(tokenizer, "Custom", "exact", [2, 5]),
                         [(2, [4, 8]), (5, [4, 8, 12, 4, 8])])
        self.assertEqual(prompt_ids(Tokenizer(), 5), [4, 8, 12, 4, 8])

    def test_cli_validation(self):
        parser = make_parser()
        self.assertEqual(read_prompt(parser.parse_args(["--prompt", "Hello"])), ("Hello", "text"))
        with self.assertRaises(ValueError):
            read_prompt(parser.parse_args(["--prompt", "   "]))
        self.assertEqual(parser.parse_args([]).gammas, [4])
        self.assertEqual(parser.parse_args(["--gammas", "2", "4", "6"]).gammas, [2, 4, 6])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--gammas", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--prompt", "a", "--interactive"])

    def test_plot_fixture_and_incomplete_rejection(self):
        # Artificial values exercise plotting; they are not inference results.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic_fixture.csv"
            rows = []
            for n in [128]:
                for method, gamma in [("baseline", ""), ("speculative", 2), ("speculative", 4)]:
                    for repeat, value in [(1, 10.0), (2, 20.0)]:
                        rows.append(dict(
                            experiment_id="synthetic-test", draft_model="draft-fixture",
                            target_model="target-fixture", prompt_tokens=n, method=method, gamma=gamma,
                            repeat=repeat, repeats=2, prompt_lengths="[128]", gammas="[2, 4]",
                            tokens_match=True, output_tokens_per_second=value,
                            acceptance_rate=(0.8 if method == "speculative" else ""),
                            mean_committed_per_round=(3.0 if method == "speculative" else 1.0),
                            peak_allocated_mib=value))

            def save(items):
                with source.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(items)

            command = [sys.executable, str(Path(__file__).with_name("plot_results.py")),
                       str(source), "--output-dir", str(root / "plots")]
            save(rows)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((root / "plots" / "benchmark.png").stat().st_size, 1000)

            save(rows[:-1])  # drop one measured run -> incomplete experiment
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
