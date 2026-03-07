"""
evaluate.py — Perplexity + ROUGE-L evaluation.

Two metrics:
  1. Perplexity (PPL) — measures how surprised the model is by the held-out text.
       PPL = exp(mean_cross_entropy_loss)
     Lower is better. A random model on 32K vocab ≈ 32,000; a good 7B SFT ≈ 5-15.

  2. ROUGE-L — longest-common-subsequence recall between reference and generated text.
     Measures surface-level sequence overlap. 0–1 (higher is better).

Usage
-----
    python -m src.evaluate \\
        --adapter  output/run/final \\
        --base     meta-llama/Meta-Llama-3-8B \\
        --dataset  tatsu-lab/alpaca \\
        --split    test \\
        --n-rouge  200
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L (pure Python, no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Dynamic-programming LCS length on token lists."""
    m, n = len(a), len(b)
    # Space-optimised: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """
    Compute sentence-level ROUGE-L F1.

    Tokenises on whitespace (quick, language-agnostic).
    For production use, replace with `rouge_score` library for proper stemming.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def corpus_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """Average ROUGE-L F1 over a corpus."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    scores = [rouge_l_score(r, h) for r, h in zip(references, hypotheses)]
    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    dataloader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity on a DataLoader.

    PPL = exp( (1/N) * Σ cross_entropy_loss_i )

    Parameters
    ----------
    model      : any HuggingFace CausalLM (or PEFT-wrapped version)
    dataloader : yields batches with keys input_ids / attention_mask / labels
    device     : if None, inferred from model parameters

    Returns
    -------
    float  Perplexity (exp of mean NLL)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # outputs.loss is already mean NLL over non-masked tokens
            loss = outputs.loss
            # count non-masked label tokens in this batch
            labels = batch["labels"]
            n_tokens = (labels != -100).sum().item()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# Generation + ROUGE-L evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = False          # greedy by default for deterministic eval


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    gen_cfg: Optional[GenerationConfig] = None,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate model completions for a list of prompts."""
    if gen_cfg is None:
        gen_cfg = GenerationConfig()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    results: List[str] = []

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature if gen_cfg.do_sample else 1.0,
                top_p=gen_cfg.top_p if gen_cfg.do_sample else 1.0,
                repetition_penalty=gen_cfg.repetition_penalty,
                do_sample=gen_cfg.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = enc["input_ids"].shape[1]
        for ids in out_ids:
            new_ids = ids[input_len:]
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(decoded.strip())

    return results


def evaluate_rouge(
    model,
    tokenizer,
    examples: List[Dict],          # list of {"prompt": ..., "reference": ...}
    gen_cfg: Optional[GenerationConfig] = None,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Generate completions and compute ROUGE-L vs reference outputs.

    Returns
    -------
    dict with keys:
      "rouge_l"   : corpus-level ROUGE-L F1
      "n_samples" : number of evaluated samples
    """
    prompts = [ex["prompt"] for ex in examples]
    references = [ex["reference"] for ex in examples]

    logger.info("Generating %d responses for ROUGE-L evaluation...", len(prompts))
    hypotheses = generate_responses(
        model, tokenizer, prompts, gen_cfg=gen_cfg,
        batch_size=batch_size, device=device,
    )

    rouge = corpus_rouge_l(references, hypotheses)
    logger.info("ROUGE-L: %.4f  (n=%d)", rouge, len(prompts))
    return {"rouge_l": rouge, "n_samples": len(prompts)}


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    base_model_name: str,
    adapter_path: Optional[str],
    dataset_name: str = "tatsu-lab/alpaca",
    dataset_format: str = "alpaca",
    split: str = "test",
    n_rouge_samples: int = 200,
    max_length: int = 2048,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Load (optionally fine-tuned) model, run perplexity + ROUGE-L, return metrics.
    """
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .dataset import (
        DatasetConfig,
        FineTuningDataset,
        DataCollatorForSeq2Seq,
        _get_formatter,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapters from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # ── Perplexity dataset ───────────────────────────────────────────────────
    ds_cfg = DatasetConfig(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
        split=split,
        max_length=max_length,
    )
    ppl_ds = FineTuningDataset.from_config(ds_cfg, tokenizer, split=split)
    collator = DataCollatorForSeq2Seq(tokenizer)
    ppl_loader = DataLoader(ppl_ds, batch_size=batch_size, collate_fn=collator)

    logger.info("Computing perplexity on %d examples...", len(ppl_ds))
    ppl = compute_perplexity(model, ppl_loader, device=device)
    logger.info("Perplexity: %.2f", ppl)

    # ── ROUGE-L ──────────────────────────────────────────────────────────────
    from datasets import load_dataset as hf_load

    raw = hf_load(dataset_name, split=split)
    formatter = _get_formatter(dataset_format)

    rouge_examples = []
    for ex in raw:
        if len(rouge_examples) >= n_rouge_samples:
            break
        prompt, reference = formatter(ex)
        if reference:
            rouge_examples.append({"prompt": prompt, "reference": reference})

    rouge_metrics = evaluate_rouge(
        model, tokenizer, rouge_examples, batch_size=batch_size, device=device
    )

    metrics = {"perplexity": ppl, **rouge_metrics}
    logger.info("Final metrics: %s", metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Evaluate fine-tuned model (PPL + ROUGE-L)")
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (optional)")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--format", default="alpaca", choices=["alpaca", "sharegpt"])
    p.add_argument("--split", default="test")
    p.add_argument("--n-rouge", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    metrics = run_evaluation(
        base_model_name=args.base,
        adapter_path=args.adapter,
        dataset_name=args.dataset,
        dataset_format=args.format,
        split=args.split,
        n_rouge_samples=args.n_rouge,
        batch_size=args.batch_size,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
