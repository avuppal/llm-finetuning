"""
tests/test_dataset.py

Tests for dataset.py:
  - Alpaca prompt formatting
  - ShareGPT prompt formatting
  - Tokenisation + loss masking
  - Truncation behaviour
  - DataCollatorForSeq2Seq padding
  - FineTuningDataset.from_list (no HF hub access required)
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class MockTokenizer:
    """Minimal tokenizer that encodes text as character-level token ids."""

    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, text, max_length=2048, truncation=True, padding=False,
                 return_tensors=None, **kwargs):
        if isinstance(text, list):
            # batch mode
            encoded = [self._encode_single(t, max_length, truncation) for t in text]
            max_len = max(len(e) for e in encoded)
            padded = [e + [0] * (max_len - len(e)) for e in encoded]
            masks = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded]
            result = MagicMock()
            if return_tensors == "pt":
                result.__getitem__ = lambda self, k: (
                    torch.tensor(padded) if k == "input_ids" else torch.tensor(masks)
                )
                result.to = lambda device: result
            return result
        else:
            ids = self._encode_single(text, max_length, truncation)
            result = MagicMock()
            if return_tensors == "pt":
                result.__getitem__ = lambda self, k: (
                    torch.tensor([ids]) if k == "input_ids" else torch.ones(1, len(ids), dtype=torch.long)
                )
            return result

    def _encode_single(self, text, max_length, truncation):
        ids = [ord(c) % 256 for c in text]
        if truncation:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(max(32, i % 128)) for i in ids)


@pytest.fixture
def tokenizer():
    return MockTokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca formatting
# ─────────────────────────────────────────────────────────────────────────────

def test_format_alpaca_with_input():
    from src.dataset import format_alpaca
    ex = {
        "instruction": "Translate to French",
        "input": "Hello, world!",
        "output": "Bonjour, monde!",
    }
    prompt, completion = format_alpaca(ex)
    assert "Translate to French" in prompt
    assert "Hello, world!" in prompt
    assert "### Response:" in prompt
    assert completion == "Bonjour, monde!"


def test_format_alpaca_without_input():
    from src.dataset import format_alpaca
    ex = {
        "instruction": "Write a haiku about spring",
        "input": "",
        "output": "Cherry blossoms fall\nSilently to the cool ground\nSpring has said goodbye",
    }
    prompt, completion = format_alpaca(ex)
    assert "Write a haiku" in prompt
    assert "further context" not in prompt.lower() or "### Input:" not in prompt
    assert "### Response:" in prompt
    assert completion.startswith("Cherry")


def test_format_alpaca_missing_keys():
    from src.dataset import format_alpaca
    ex = {"instruction": "Do something", "output": "Result"}
    prompt, completion = format_alpaca(ex)
    assert "Do something" in prompt
    assert completion == "Result"


# ─────────────────────────────────────────────────────────────────────────────
# ShareGPT formatting
# ─────────────────────────────────────────────────────────────────────────────

def test_format_sharegpt_basic():
    from src.dataset import format_sharegpt
    ex = {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    prompt, completion = format_sharegpt(ex)
    assert "What is 2+2?" in prompt
    assert "Assistant:" in prompt
    assert completion == "4"


def test_format_sharegpt_multi_turn():
    from src.dataset import format_sharegpt
    ex = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
            {"from": "human", "value": "What's the capital of France?"},
            {"from": "gpt", "value": "Paris"},
        ]
    }
    prompt, completion = format_sharegpt(ex)
    assert "Hi" in prompt
    assert "Hello!" in prompt
    assert "capital of France" in prompt
    assert completion == "Paris"
    # Last assistant message should NOT be in prompt
    assert "Paris" not in prompt


def test_format_sharegpt_empty():
    from src.dataset import format_sharegpt
    ex = {"conversations": []}
    prompt, completion = format_sharegpt(ex)
    assert prompt == ""
    assert completion == ""


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation + loss masking
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenize_returns_required_keys(tokenizer):
    from src.dataset import tokenize_with_loss_mask
    result = tokenize_with_loss_mask(tokenizer, "Instruction: do X\n### Response:\n", "The answer is X.")
    assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)


def test_loss_mask_prompt_tokens_are_minus100(tokenizer):
    from src.dataset import tokenize_with_loss_mask
    prompt = "Below is an instruction:\n### Instruction:\nTranslate.\n### Response:\n"
    completion = "Translation here."
    result = tokenize_with_loss_mask(tokenizer, prompt, completion, max_length=512)
    labels = result["labels"]
    # At least some labels must be -100 (the prompt part)
    assert (labels == -100).any(), "Expected some prompt tokens masked with -100"
    # At least some labels must NOT be -100 (the completion part)
    assert (labels != -100).any(), "Expected some completion tokens with real labels"


def test_loss_mask_values_match_input_ids(tokenizer):
    from src.dataset import tokenize_with_loss_mask
    result = tokenize_with_loss_mask(tokenizer, "Q: Hello\nA: ", "World", max_length=256)
    ids = result["input_ids"]
    labels = result["labels"]
    # Non-masked positions should equal input_ids
    non_masked = labels != -100
    assert torch.all(ids[non_masked] == labels[non_masked])


def test_truncation_respects_max_length(tokenizer):
    from src.dataset import tokenize_with_loss_mask
    long_prompt = "A" * 200
    long_completion = "B" * 200
    result = tokenize_with_loss_mask(tokenizer, long_prompt, long_completion, max_length=100)
    assert result["input_ids"].shape[0] <= 100
    assert result["labels"].shape[0] <= 100


# ─────────────────────────────────────────────────────────────────────────────
# FineTuningDataset
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_from_list_alpaca(tokenizer):
    from src.dataset import FineTuningDataset
    records = [
        {"instruction": "Say hi", "input": "", "output": "Hi!"},
        {"instruction": "Say bye", "input": "", "output": "Bye!"},
    ]
    ds = FineTuningDataset.from_list(records, tokenizer, fmt="alpaca")
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item


def test_dataset_from_list_sharegpt(tokenizer):
    from src.dataset import FineTuningDataset
    records = [
        {
            "conversations": [
                {"from": "human", "value": "How are you?"},
                {"from": "gpt", "value": "I am fine, thanks!"},
            ]
        }
    ]
    ds = FineTuningDataset.from_list(records, tokenizer, fmt="sharegpt")
    assert len(ds) == 1


def test_dataset_skips_empty_completion(tokenizer):
    from src.dataset import FineTuningDataset
    records = [
        {"instruction": "Valid", "input": "", "output": "Good"},
        {"instruction": "Missing output", "input": "", "output": ""},
    ]
    ds = FineTuningDataset.from_list(records, tokenizer, fmt="alpaca")
    # The empty-output example should be filtered out
    assert len(ds) == 1


def test_dataset_unknown_format_raises(tokenizer):
    from src.dataset import _get_formatter
    with pytest.raises(ValueError, match="Unknown dataset format"):
        _get_formatter("csv")


# ─────────────────────────────────────────────────────────────────────────────
# DataCollatorForSeq2Seq
# ─────────────────────────────────────────────────────────────────────────────

def test_collator_pads_to_max_length(tokenizer):
    from src.dataset import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=0)

    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([- 100, 2, 3]),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 5]),
        },
    ]

    out = collator(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    # Padding: second example padded with 0 (pad_token_id)
    assert out["input_ids"][1, 2].item() == 0
    assert out["attention_mask"][1, 2].item() == 0
    assert out["labels"][1, 2].item() == -100


def test_collator_pad_to_multiple(tokenizer):
    from src.dataset import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, 3, 4, 5]),
        }
    ]
    out = collator(batch)
    seq_len = out["input_ids"].shape[1]
    assert seq_len % 8 == 0
    assert seq_len >= 5
