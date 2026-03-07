"""
dataset.py — HuggingFace dataset loading + prompt formatting.

Supports:
  - Alpaca format  : {"instruction": ..., "input": ..., "output": ...}
  - ShareGPT format: {"conversations": [{"from": "human"|"gpt", "value": ...}]}

Returns tokenized tensors with:
  - input_ids       : token ids
  - attention_mask  : 1 for real tokens, 0 for padding
  - labels          : same as input_ids but -100 on prompt tokens (no loss on prompt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_alpaca(example: Dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, completion) for an Alpaca-format example."""
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if inp:
        prompt = ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    else:
        prompt = ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)

    return prompt, output


def format_sharegpt(example: Dict[str, Any]) -> tuple[str, str]:
    """
    Return (prompt, completion) for a ShareGPT-format example.

    ShareGPT conversations alternate human/gpt turns.
    We treat everything up to and including the last human turn as the prompt,
    and the final gpt turn as the completion.
    """
    conversations: List[Dict[str, str]] = example.get("conversations", [])

    if not conversations:
        return "", ""

    # Build turn-by-turn text; split at the last assistant response
    prompt_parts: List[str] = []
    completion: str = ""

    for i, turn in enumerate(conversations):
        role = turn.get("from", "").lower()
        value = turn.get("value", "").strip()

        if role in ("human", "user"):
            prompt_parts.append(f"Human: {value}\n")
        elif role in ("gpt", "assistant"):
            if i == len(conversations) - 1:
                # last turn → completion
                completion = value
            else:
                prompt_parts.append(f"Assistant: {value}\n")

    prompt = "".join(prompt_parts) + "Assistant: "
    return prompt, completion


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_with_loss_mask(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompt+completion and set labels=-100 on prompt tokens.

    Steps
    -----
    1. Tokenize full text (prompt + completion + EOS).
    2. Tokenize prompt-only to find the boundary.
    3. Mask labels for all prompt positions.

    Returns dict with keys: input_ids, attention_mask, labels.
    """
    full_text = prompt + completion
    eos = tokenizer.eos_token or ""

    # Full sequence
    full_enc = tokenizer(
        full_text + eos,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = full_enc["input_ids"][0]
    attention_mask = full_enc["attention_mask"][0]

    # Prompt-only length (to find the mask boundary)
    prompt_enc = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    prompt_len = prompt_enc["input_ids"].shape[1]
    # clamp in case truncation made full_seq shorter than prompt
    prompt_len = min(prompt_len, len(input_ids))

    labels = input_ids.clone()
    labels[:prompt_len] = -100  # mask prompt tokens → no loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    dataset_name: str = "tatsu-lab/alpaca"          # HF hub name or local path
    dataset_format: str = "alpaca"                   # "alpaca" | "sharegpt"
    split: str = "train"
    max_length: int = 2048
    val_split_ratio: float = 0.05                    # fraction held out for eval
    seed: int = 42
    num_proc: int = 4
    streaming: bool = False
    text_column: Optional[str] = None                # override for raw-text datasets


class FineTuningDataset(Dataset):
    """
    PyTorch Dataset wrapping a HuggingFace dataset.

    Usage
    -----
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    >>> ds = FineTuningDataset.from_config(DatasetConfig(), tokenizer)
    """

    def __init__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: DatasetConfig,
        tokenizer,
        split: Optional[str] = None,
    ) -> "FineTuningDataset":
        """Load, format, and tokenize a dataset from config."""
        from datasets import load_dataset  # local import — optional dep

        split = split or config.split
        logger.info("Loading dataset '%s' split='%s'", config.dataset_name, split)

        raw = load_dataset(
            config.dataset_name,
            split=split,
            streaming=config.streaming,
        )

        formatter = _get_formatter(config.dataset_format)
        examples: List[Dict[str, torch.Tensor]] = []

        for ex in raw:
            prompt, completion = formatter(ex)
            if not completion:
                continue
            tokenized = tokenize_with_loss_mask(
                tokenizer, prompt, completion, max_length=config.max_length
            )
            examples.append(tokenized)

        logger.info("Tokenized %d examples", len(examples))
        return cls(examples)

    @classmethod
    def from_list(
        cls,
        records: List[Dict[str, Any]],
        tokenizer,
        fmt: str = "alpaca",
        max_length: int = 2048,
    ) -> "FineTuningDataset":
        """Convenience constructor for in-memory lists (useful in tests)."""
        formatter = _get_formatter(fmt)
        examples: List[Dict[str, torch.Tensor]] = []
        for ex in records:
            prompt, completion = formatter(ex)
            if not completion:
                continue
            tokenized = tokenize_with_loss_mask(
                tokenizer, prompt, completion, max_length=max_length
            )
            examples.append(tokenized)
        return cls(examples)


def _get_formatter(fmt: str):
    fmt = fmt.lower()
    if fmt == "alpaca":
        return format_alpaca
    elif fmt == "sharegpt":
        return format_sharegpt
    else:
        raise ValueError(f"Unknown dataset format: '{fmt}'. Choose 'alpaca' or 'sharegpt'.")


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class DataCollatorForSeq2Seq:
    """
    Pads input_ids / attention_mask / labels to the longest sequence in a batch.

    Labels are padded with -100 so padding positions are ignored by the loss.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        max_len = max(ex["input_ids"].shape[0] for ex in batch)
        # round up to multiple for tensor-core efficiency
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ex in batch:
            seq_len = ex["input_ids"].shape[0]
            pad_len = max_len - seq_len

            padded_input_ids.append(
                torch.cat([ex["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            padded_attention_mask.append(
                torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([ex["labels"], torch.full((pad_len,), -100)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
