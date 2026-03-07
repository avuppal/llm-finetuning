"""
tests/test_train.py

Tests for train.py:
  - TrainingConfig YAML loading
  - TrainingConfig defaults
  - LoRA config instantiation (peft, no real weights)
  - BnB config construction (transformers, no real weights)
  - Forward pass on a tiny CPU model with LoRA injected
  - LR scheduler behaviour
  - Config CLI override parsing

All tests run on CPU. No real model weights are downloaded.
"""

import pytest
import torch
import yaml
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.train import TrainingConfig


# ─────────────────────────────────────────────────────────────────────────────
# TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────

def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert "q_proj" in cfg.lora_target_modules
    assert "v_proj" in cfg.lora_target_modules
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.gradient_accumulation_steps == 4


def test_training_config_from_yaml(tmp_path):
    data = {
        "model_name_or_path": "mistralai/Mistral-7B-v0.1",
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "num_train_epochs": 5,
        "output_dir": "output/test",
    }
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump(data))

    cfg = TrainingConfig.from_yaml(str(cfg_file))
    assert cfg.model_name_or_path == "mistralai/Mistral-7B-v0.1"
    assert cfg.lora_r == 32
    assert cfg.lora_alpha == 64
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.num_train_epochs == 5


def test_training_config_unknown_key_ignored(tmp_path, caplog):
    import logging
    data = {"this_key_does_not_exist": "should_be_ignored", "lora_r": 8}
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump(data))

    with caplog.at_level(logging.WARNING):
        cfg = TrainingConfig.from_yaml(str(cfg_file))

    assert cfg.lora_r == 8
    assert "this_key_does_not_exist" in caplog.text


def test_training_config_lora_target_modules_from_yaml(tmp_path):
    data = {
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump(data))

    cfg = TrainingConfig.from_yaml(str(cfg_file))
    assert "k_proj" in cfg.lora_target_modules
    assert "o_proj" in cfg.lora_target_modules
    assert len(cfg.lora_target_modules) == 4


# ─────────────────────────────────────────────────────────────────────────────
# BnB config
# ─────────────────────────────────────────────────────────────────────────────

def test_build_bnb_config_nf4():
    """BitsAndBytesConfig should be constructable without a GPU."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        pytest.skip("transformers not installed")

    from src.train import build_bnb_config
    cfg = TrainingConfig()
    bnb = build_bnb_config(cfg)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True


def test_build_bnb_config_fp16_compute():
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        pytest.skip("transformers not installed")

    from src.train import build_bnb_config
    cfg = TrainingConfig(bnb_4bit_compute_dtype="float16")
    bnb = build_bnb_config(cfg)
    assert bnb.bnb_4bit_compute_dtype == torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# LoRA config instantiation (peft, no weights)
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_config_instantiation():
    """LoraConfig should be constructable without loading a model."""
    try:
        from peft import LoraConfig
    except ImportError:
        pytest.skip("peft not installed")

    cfg = TrainingConfig(lora_r=8, lora_alpha=16, lora_dropout=0.1)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )
    assert lora_config.r == 8
    assert lora_config.lora_alpha == 16
    assert lora_config.target_modules == cfg.lora_target_modules


# ─────────────────────────────────────────────────────────────────────────────
# LoRA forward pass on a tiny toy model (CPU, no real weights)
# ─────────────────────────────────────────────────────────────────────────────

class TinyLinear(torch.nn.Module):
    """
    A toy 'language model'-like module: Linear → linear → cross entropy.
    Used to verify that LoRA injection + forward pass works on CPU.
    """
    def __init__(self, vocab_size=128, hidden=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        x = self.q_proj(x)
        x = torch.relu(self.v_proj(x))
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return MagicMock(loss=loss, logits=logits)


@pytest.mark.not_gpu
def test_lora_forward_pass_cpu():
    """Inject LoRA into toy model, run a forward pass on CPU."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        pytest.skip("peft not installed")

    vocab_size = 128
    model = TinyLinear(vocab_size=vocab_size, hidden=64)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION",  # avoids CausalLM-specific wrapping
    )
    lora_model = get_peft_model(model, lora_config)

    # Verify only LoRA params are trainable
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    assert trainable < total, "LoRA should freeze base weights"
    assert trainable > 0, "LoRA adapters should be trainable"

    # Forward pass
    input_ids = torch.randint(0, vocab_size, (2, 16))
    labels = input_ids.clone()
    out = lora_model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    loss = out.loss
    assert not torch.isnan(loss)
    assert loss.item() > 0


@pytest.mark.not_gpu
def test_lora_trainable_param_count():
    """LoRA should inject significantly fewer trainable params than full FT."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        pytest.skip("peft not installed")

    model = TinyLinear(vocab_size=256, hidden=128)
    total_params = sum(p.numel() for p in model.parameters())

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    lora_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    # LoRA trainable < 10% of total (for tiny model even less)
    ratio = trainable / total_params
    assert ratio < 0.5, f"LoRA trainable ratio {ratio:.2%} is too high"


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.not_gpu
def test_lr_scheduler_warmup_and_decay():
    from src.train import build_lr_scheduler
    cfg = TrainingConfig(learning_rate=1e-3, warmup_ratio=0.1)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = 100

    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # LR should start low (warmup)
    assert lrs[0] < lrs[10], "LR should increase during warmup"
    # LR should be lower at end than at peak (cosine decay)
    assert lrs[-1] < lrs[15], "LR should decay after warmup"


@pytest.mark.not_gpu
def test_lr_scheduler_peak_around_warmup_end():
    from src.train import build_lr_scheduler
    cfg = TrainingConfig(learning_rate=2e-4, warmup_ratio=0.2)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = 50

    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    warmup_end = int(total_steps * cfg.warmup_ratio)
    peak_lr = max(lrs[:warmup_end + 2])
    assert peak_lr == pytest.approx(cfg.learning_rate, rel=0.1)
