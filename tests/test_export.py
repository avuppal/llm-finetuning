"""
tests/test_export.py

Tests for export.py — all run on CPU with no real model weights.

The strategy for mocking transformers/peft:
  export.py uses *lazy imports* (imports inside the function body).
  When peft is first imported it tries to import transformers.PreTrainedModel,
  which fails on this environment (version conflict).

  Fix: inject lightweight fake modules into sys.modules BEFORE any code path
  that would trigger the real import.  We do this once at module load time
  so every test in this file sees clean mocks.
"""

import json
import sys
import types
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# sys.modules injection — must run before any `from src.export import …`
# ─────────────────────────────────────────────────────────────────────────────

def _inject_fake_modules() -> None:
    """
    Inject minimal fake modules for transformers and peft so that export.py's
    lazy imports resolve to mocks without triggering the real package's broken
    top-level import chain.
    """
    # Build a fake 'transformers' module with the attributes export.py needs
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = MagicMock()
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.pipeline = MagicMock()

    # Build a fake 'peft' module
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = MagicMock()

    sys.modules.setdefault("transformers", fake_transformers)
    sys.modules.setdefault("peft", fake_peft)


_inject_fake_modules()


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L (pure-Python implementation in evaluate.py — no torch/HF needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_rouge_l_identical():
    from src.evaluate import rouge_l_score
    assert rouge_l_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_rouge_l_disjoint():
    from src.evaluate import rouge_l_score
    assert rouge_l_score("apple orange banana", "xyz foo bar") == pytest.approx(0.0)


def test_rouge_l_partial():
    from src.evaluate import rouge_l_score
    score = rouge_l_score("a b c d", "a b x y")
    assert 0.0 < score < 1.0


def test_rouge_l_empty_reference():
    from src.evaluate import rouge_l_score
    assert rouge_l_score("", "some hypothesis") == 0.0


def test_rouge_l_empty_hypothesis():
    from src.evaluate import rouge_l_score
    assert rouge_l_score("some reference", "") == 0.0


def test_corpus_rouge_l_average():
    from src.evaluate import corpus_rouge_l
    assert corpus_rouge_l(["a b c", "x y z"], ["a b c", "x y z"]) == pytest.approx(1.0)


def test_corpus_rouge_l_length_mismatch_raises():
    from src.evaluate import corpus_rouge_l
    with pytest.raises(ValueError, match="same length"):
        corpus_rouge_l(["a"], ["b", "c"])


def test_corpus_rouge_l_empty():
    from src.evaluate import corpus_rouge_l
    assert corpus_rouge_l([], []) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity computation (mocked model)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_model(loss_value: float = 2.0):
    model = MagicMock()
    out = MagicMock()
    out.loss = torch.tensor(loss_value)
    model.return_value = out
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    return model


def test_compute_perplexity_known_loss():
    import math
    from src.evaluate import compute_perplexity

    batch = {
        "input_ids":      torch.ones(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels":         torch.ones(2, 8, dtype=torch.long),
    }
    ppl = compute_perplexity(_make_mock_model(2.0), [batch], device=torch.device("cpu"))
    assert ppl == pytest.approx(math.exp(2.0), rel=1e-4)


def test_compute_perplexity_masked_labels():
    import math
    from src.evaluate import compute_perplexity

    labels = torch.full((2, 8), -100, dtype=torch.long)
    labels[:, 4:] = 1  # 8 non-masked tokens
    batch = {
        "input_ids":      torch.ones(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels":         labels,
    }
    ppl = compute_perplexity(_make_mock_model(3.0), [batch], device=torch.device("cpu"))
    assert ppl == pytest.approx(math.exp(3.0), rel=1e-4)


def test_compute_perplexity_empty_loader_returns_inf():
    from src.evaluate import compute_perplexity
    assert compute_perplexity(_make_mock_model(), [], device=torch.device("cpu")) == float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# merge_lora_into_base (mocked via sys.modules stubs)
# ─────────────────────────────────────────────────────────────────────────────

def _make_merge_mocks():
    """Populate the fake sys.modules stubs with per-test mocks and return them."""
    fake_transformers = sys.modules["transformers"]
    fake_peft         = sys.modules["peft"]

    # Merged model returned by merge_and_unload()
    merged = MagicMock()
    merged.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    # Base model
    base_model = MagicMock()
    base_model.config = MagicMock()

    # PEFT model wrapping the base
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged

    # Tokenizer
    tokenizer = MagicMock()

    # Wire up factory methods on the fake modules
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = base_model
    fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    fake_peft.PeftModel.from_pretrained.return_value = peft_model

    return base_model, peft_model, merged, tokenizer


def test_merge_lora_creates_output_dir():
    """merge_lora_into_base must create the output directory."""
    _make_merge_mocks()
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "merged"
        from src.export import merge_lora_into_base
        result = merge_lora_into_base(
            base_model_name_or_path="fake/model",
            adapter_path="fake/adapter",
            output_dir=str(output_dir),
            device="cpu",
        )
        assert result == output_dir.resolve()
        assert output_dir.exists()


def test_merge_lora_calls_save_pretrained():
    """save_pretrained must be called on the merged model and the tokenizer."""
    _, _, merged, tokenizer = _make_merge_mocks()
    with tempfile.TemporaryDirectory() as tmp:
        from src.export import merge_lora_into_base
        merge_lora_into_base(
            base_model_name_or_path="fake/model",
            adapter_path="fake/adapter",
            output_dir=str(Path(tmp) / "out"),
            device="cpu",
        )
    merged.save_pretrained.assert_called_once()
    tokenizer.save_pretrained.assert_called_once()


def test_merge_lora_merge_and_unload_called():
    """merge_and_unload() must be called exactly once."""
    _, peft_model, _, _ = _make_merge_mocks()
    with tempfile.TemporaryDirectory() as tmp:
        from src.export import merge_lora_into_base
        merge_lora_into_base(
            base_model_name_or_path="fake/model",
            adapter_path="fake/adapter",
            output_dir=str(Path(tmp) / "out"),
            device="cpu",
        )
    peft_model.merge_and_unload.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# save_export_manifest
# ─────────────────────────────────────────────────────────────────────────────

def test_save_export_manifest_creates_file():
    from src.export import save_export_manifest
    with tempfile.TemporaryDirectory() as tmp:
        save_export_manifest(
            output_dir=tmp,
            base_model="meta-llama/Meta-Llama-3-8B",
            adapter_path="output/run/final",
            dtype="float16",
            verified=True,
            prompt_response="LoRA is great.",
        )
        assert (Path(tmp) / "export_manifest.json").exists()


def test_save_export_manifest_content():
    from src.export import save_export_manifest
    with tempfile.TemporaryDirectory() as tmp:
        save_export_manifest(
            output_dir=tmp,
            base_model="my/base",
            adapter_path="my/adapter",
            dtype="bfloat16",
            verified=False,
        )
        with open(Path(tmp) / "export_manifest.json") as f:
            data = json.load(f)
    assert data["base_model"] == "my/base"
    assert data["adapter_path"] == "my/adapter"
    assert data["export_dtype"] == "bfloat16"
    assert data["verified"] is False
    assert data["format"] == "safetensors"
    assert "exported_at" in data
