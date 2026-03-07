"""
export.py — Merge LoRA adapters into the base model and export as safetensors.

Pipeline
--------
1. Load base model (float16 / bfloat16, CPU-friendly for export)
2. Load LoRA adapters via PEFT
3. Call .merge_and_unload()  → fuses A·B into each W, removes adapter layers
4. Save merged model with save_pretrained (safetensors by default)
5. Verify: reload the exported model and run a smoke-test prompt

Why merge before serving?
  - Eliminates runtime LoRA overhead (no extra matmuls during inference)
  - Makes the model compatible with ANY inference runtime (vLLM, llama.cpp, TGI, etc.)
  - Required for GGUF quantisation

Usage
-----
    python -m src.export \\
        --base   meta-llama/Meta-Llama-3-8B \\
        --adapter output/run/final \\
        --output  merged_model/ \\
        --verify  "Write a poem about neural networks."
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core merge logic
# ─────────────────────────────────────────────────────────────────────────────

def merge_lora_into_base(
    base_model_name_or_path: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
) -> Path:
    """
    Merge LoRA weights into the base model and save as safetensors.

    Parameters
    ----------
    base_model_name_or_path : HF model id or local path to base model
    adapter_path            : path to saved PEFT adapter (from train.py output)
    output_dir              : where to write the merged model
    torch_dtype             : dtype for the merged weights (fp16 for serving, bf16 for training)
    device                  : "cpu" is safer for large models; "cuda" is faster
    push_to_hub             : if True, also push to HuggingFace Hub
    hub_repo_id             : e.g. "avuppal/llama3-8b-merged"

    Returns
    -------
    Path  absolute path of the output directory
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load base model ───────────────────────────────────────────────────
    logger.info("Loading base model: %s  (dtype=%s, device=%s)",
                base_model_name_or_path, torch_dtype, device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = True  # re-enable for inference

    # ── 2. Load LoRA adapters ────────────────────────────────────────────────
    logger.info("Loading LoRA adapters from: %s", adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
    )

    # ── 3. Merge + unload ────────────────────────────────────────────────────
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    #   After merge_and_unload():
    #   - Every target Linear has W_new = W_frozen + (lora_B @ lora_A) * (alpha/r)
    #   - All LoRA-specific parameters and hooks are removed
    #   - model is a plain AutoModelForCausalLM again

    logger.info(
        "Merge complete. Param count: %s M",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}",
    )

    # ── 4. Save as safetensors ───────────────────────────────────────────────
    logger.info("Saving merged model → %s", output_path)
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,    # safetensors format (no pickle)
        max_shard_size="5GB",       # shard large models for Hub compatibility
    )

    # Save tokenizer alongside the weights
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_path))
    logger.info("Tokenizer saved → %s", output_path)

    # ── 5. Optionally push to Hub ────────────────────────────────────────────
    if push_to_hub and hub_repo_id:
        logger.info("Pushing to HuggingFace Hub: %s", hub_repo_id)
        model.push_to_hub(hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_merged_model(
    output_dir: str,
    prompt: str = "Explain the key idea behind LoRA in one paragraph.",
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> str:
    """
    Reload the exported model from disk and run a test prompt.

    This catches common export failures:
      - Corrupted safetensors shards
      - Missing tokenizer config
      - Weight dtype mismatch
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    logger.info("Verifying exported model from: %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    output = pipe(prompt)[0]["generated_text"]
    # Strip the prompt prefix
    completion = output[len(prompt):].strip()

    logger.info("Smoke-test prompt   : %s", prompt)
    logger.info("Smoke-test response : %s", completion[:200] + "..." if len(completion) > 200 else completion)
    return completion


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: save adapter metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_export_manifest(
    output_dir: str,
    base_model: str,
    adapter_path: str,
    dtype: str,
    verified: bool,
    prompt_response: Optional[str] = None,
) -> None:
    """Write a JSON manifest alongside the exported weights for reproducibility."""
    import json
    from datetime import datetime, timezone

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter_path": adapter_path,
        "export_dtype": dtype,
        "format": "safetensors",
        "verified": verified,
        "verification_response": prompt_response,
    }

    manifest_path = Path(output_dir) / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written → %s", manifest_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Merge LoRA adapters and export safetensors")
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", required=True, help="Path to saved LoRA adapters")
    p.add_argument("--output", required=True, help="Output directory for merged model")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for exported weights",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for export ('cpu' or 'cuda')",
    )
    p.add_argument(
        "--verify",
        default=None,
        metavar="PROMPT",
        help="Run a smoke-test with this prompt after export",
    )
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo-id", default=None)
    args = p.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    output_path = merge_lora_into_base(
        base_model_name_or_path=args.base,
        adapter_path=args.adapter,
        output_dir=args.output,
        torch_dtype=dtype_map[args.dtype],
        device=args.device,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )

    prompt_response: Optional[str] = None
    verified = False
    if args.verify:
        prompt_response = verify_merged_model(
            str(output_path),
            prompt=args.verify,
            device=args.device,
        )
        verified = True
        print(f"\n=== Verification Response ===\n{prompt_response}\n")

    save_export_manifest(
        str(output_path),
        base_model=args.base,
        adapter_path=args.adapter,
        dtype=args.dtype,
        verified=verified,
        prompt_response=prompt_response,
    )

    print(f"\nExport complete → {output_path}")


if __name__ == "__main__":
    main()
