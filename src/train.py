"""
train.py — QLoRA / LoRA training loop.

Features
--------
- 4-bit NF4 quantisation via bitsandbytes
- LoRA adapters injected via PEFT (targets q_proj / v_proj by default)
- Gradient checkpointing to reduce activation memory
- Paged AdamW (bitsandbytes) for CPU-offloaded optimizer states
- YAML config-driven — all hyper-parameters live in configs/*.yaml
- Structured logging: loss / lr / grad_norm every N steps

Usage
-----
    python -m src.train --config configs/qlora_7b.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    use_flash_attention_2: bool = False

    # Quantisation
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"           # "nf4" | "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"   # "bfloat16" | "float16"
    bnb_4bit_use_double_quant: bool = True      # nested quantisation

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_bias: str = "none"

    # Dataset
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_format: str = "alpaca"
    max_length: int = 2048
    val_split_ratio: float = 0.05

    # Training
    output_dir: str = "output/run"
    num_train_epochs: int = 3
    max_steps: int = -1           # overrides epochs when > 0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    seed: int = 42

    # Checkpointing & logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "none"       # "wandb" | "tensorboard" | "none"
    run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
            else:
                logger.warning("Unknown config key '%s' — ignoring", k)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# BnB quantisation config
# ─────────────────────────────────────────────────────────────────────────────

def build_bnb_config(cfg: TrainingConfig):
    """Return a BitsAndBytesConfig for 4-bit NF4 loading."""
    from transformers import BitsAndBytesConfig

    compute_dtype = (
        torch.bfloat16
        if cfg.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(cfg: TrainingConfig):
    """
    Load a causal LM in 4-bit NF4 (QLoRA) or full precision (LoRA).

    When cfg.load_in_4bit is True:
      - Weights stored as 4-bit NF4 integers (~2× memory vs fp16)
      - Dequantised to bfloat16 for computation
      - Double-quantisation further compresses quantisation constants
    """
    from transformers import AutoModelForCausalLM

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if cfg.load_in_4bit:
        model_kwargs["quantization_config"] = build_bnb_config(cfg)
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if cfg.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info("Loading base model: %s", cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, **model_kwargs
    )
    model.config.use_cache = False          # disable KV-cache during training
    model.config.pretraining_tp = 1        # avoid tensor-parallel weight splits
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LoRA injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_lora(model, cfg: TrainingConfig):
    """
    Wrap the base model with LoRA adapters using PEFT.

    LoRA maths recap:
      W' = W + s * (B @ A)
      where  A ∈ R^(r×d),  B ∈ R^(k×r),  r << d,  s = alpha/r

    Only A and B are trainable; W is frozen.
    For QLoRA the frozen W is kept in 4-bit; A/B are in bfloat16.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit:
        # gradient checkpointing + cast layer norms to fp32
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_scheduler(optimizer, cfg: TrainingConfig, num_training_steps: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    num_warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))

    warmup = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=num_warmup_steps
    )
    main = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=cfg.learning_rate * 0.1,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[num_warmup_steps]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> None:
    """Full QLoRA training loop."""
    import random
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer

    from .dataset import DatasetConfig, FineTuningDataset, DataCollatorForSeq2Seq

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_cfg = DatasetConfig(
        dataset_name=cfg.dataset_name,
        dataset_format=cfg.dataset_format,
        max_length=cfg.max_length,
        val_split_ratio=cfg.val_split_ratio,
    )
    full_ds = FineTuningDataset.from_config(ds_cfg, tokenizer)

    val_size = max(1, int(len(full_ds) * cfg.val_split_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    collator = DataCollatorForSeq2Seq(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_base_model(cfg)
    model = inject_lora(model, cfg)

    # ── Optimizer ────────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        logger.info("Using paged AdamW (bitsandbytes)")
    except (ImportError, AttributeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        logger.info("Falling back to standard AdamW")

    # ── Steps & scheduler ────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if cfg.max_steps > 0:
        total_steps = cfg.max_steps
        num_epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        num_epochs = cfg.num_train_epochs
        total_steps = steps_per_epoch * num_epochs

    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────────
    logger.info(
        "Starting training: epochs=%d  total_steps=%d  lr=%.2e",
        num_epochs, total_steps, cfg.learning_rate,
    )

    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for micro_step, batch in enumerate(train_loader):
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break

            # Move batch to model device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + scaled loss
            outputs = model(**batch)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            # Gradient accumulation step
            if (micro_step + 1) % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % cfg.logging_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / cfg.logging_steps
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "step=%d  epoch=%d  loss=%.4f  lr=%.2e  grad_norm=%.3f  elapsed=%.1fs",
                        global_step, epoch + 1, avg_loss, current_lr,
                        grad_norm.item(), elapsed,
                    )
                    running_loss = 0.0

                # Save checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    tokenizer.save_pretrained(str(ckpt_path))
                    logger.info("Checkpoint saved → %s", ckpt_path)

        # Epoch-end eval
        val_loss = _evaluate_loss(model, val_loader)
        logger.info("Epoch %d complete — val_loss=%.4f", epoch + 1, val_loss)

    # ── Final save ───────────────────────────────────────────────────────────
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Training complete. Final adapters saved → %s", final_path)


def _evaluate_loss(model, loader) -> float:
    """Compute average cross-entropy loss on a DataLoader (no grad)."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA/LoRA fine-tuning")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override config keys, e.g. --override learning_rate=1e-4 lora_r=8",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    cfg = TrainingConfig.from_yaml(args.config)

    # Command-line overrides
    if args.override:
        for item in args.override:
            key, _, val = item.partition("=")
            if hasattr(cfg, key):
                # coerce to the existing type
                orig = getattr(cfg, key)
                try:
                    setattr(cfg, key, type(orig)(val))
                except (ValueError, TypeError):
                    setattr(cfg, key, val)
            else:
                logger.warning("Override key '%s' not found in config", key)

    train(cfg)


if __name__ == "__main__":
    main()
