"""Load a draft/target model pair that share a tokenizer, for speculative decoding.

The draft (small, fast) and target (large, accurate) models must share a
vocabulary so the target can verify the draft's raw token ids directly — this is
why both defaults come from the same model family. The tokenizer is taken from
the target, since the target defines the distribution being approximated.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type hints; the real imports stay inside the functions so that
    # `import models` works without torch/transformers installed.
    import torch
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

# Same family => identical tokenizer and vocabulary. The 0.5B drafts the 7B.
DEFAULT_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET = "Qwen/Qwen2.5-7B-Instruct"


def resolve_revision(model: str, revision: str) -> tuple[str, "PretrainedConfig"]:
    """Turn a Hub ref into a fixed commit so a run can be reproduced later. Returns
    (commit_sha, config)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, revision=revision)
    commit = getattr(config, "_commit_hash", None)
    if not commit:
        raise ValueError(f"Could not resolve a Hub commit for {model!r}; use a Hub model and revision")
    return commit, config


def load_pair(
    draft: str,
    target: str,
    draft_revision: str,
    target_revision: str,
    dtype: "torch.dtype",
    device: "torch.device",
    attn: str = "sdpa",
) -> tuple["PreTrainedModel", "PreTrainedModel", "PreTrainedTokenizerBase", dict]:
    """Load the draft and target onto `device` in `dtype`, plus the shared tokenizer.

    Returns (draft_model, target_model, tokenizer, info) where `info` carries the
    resolved commits and the target's configured context length. Raises ValueError
    if the two vocabularies differ (speculative decoding needs a shared vocab).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    draft_commit, _ = resolve_revision(draft, draft_revision)
    target_commit, target_config = resolve_revision(target, target_revision)

    tokenizer = AutoTokenizer.from_pretrained(target, revision=target_commit)

    def load(name, revision):
        return (
            AutoModelForCausalLM.from_pretrained(
                name, revision=revision, dtype=dtype, attn_implementation=attn
            )
            .to(device)
            .eval()
        )

    draft_model = load(draft, draft_commit)
    target_model = load(target, target_commit)

    if draft_model.config.vocab_size != target_model.config.vocab_size:
        raise ValueError(
            "Draft and target vocab sizes differ "
            f"({draft_model.config.vocab_size} vs {target_model.config.vocab_size}); "
            "speculative decoding requires a shared vocabulary (use models from one family)"
        )

    info = {
        "draft_revision": draft_commit,
        "target_revision": target_commit,
        "target_max_position_embeddings": target_config.max_position_embeddings,
    }
    return draft_model, target_model, tokenizer, info
