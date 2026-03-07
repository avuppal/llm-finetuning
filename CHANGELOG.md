# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet.

---

## [0.1.0] — 2026-03-07

### Added
- `src/dataset.py`: HuggingFace dataset loading with Alpaca and ShareGPT format support.
  Proper loss masking (labels=-100 on prompt tokens). `DataCollatorForSeq2Seq` with
  configurable padding to multiples of 8 for tensor-core efficiency.
- `src/train.py`: Full QLoRA training loop with:
  - 4-bit NF4 quantisation via bitsandbytes
  - LoRA injection via PEFT (`q_proj` / `v_proj` by default)
  - Gradient checkpointing for activation memory reduction
  - Paged AdamW optimizer (bitsandbytes)
  - Cosine LR scheduler with linear warmup
  - YAML-driven config (`TrainingConfig.from_yaml`)
  - Structured logging: loss / lr / grad_norm every N steps
  - CLI with `--override` support for ad-hoc hyperparameter changes
- `src/evaluate.py`: 
  - Perplexity: `exp(mean cross-entropy NLL)` on held-out split
  - ROUGE-L: pure-Python LCS-based implementation (no extra deps)
  - Generation evaluation with configurable sampling parameters
  - `run_evaluation()` pipeline function
- `src/export.py`:
  - `merge_lora_into_base()`: `merge_and_unload()` → safetensors shards
  - `verify_merged_model()`: smoke-test reload + prompt
  - `save_export_manifest()`: JSON provenance file
- `configs/qlora_7b.yaml`: Production config for LLaMA-3 8B / Mistral 7B (12 GB VRAM)
- `configs/lora_1b.yaml`: Fast-iteration config for 1B models (CPU / 4 GB GPU)
- `tests/test_dataset.py`: 16 CPU-only tests covering formatting, tokenisation, masking, collation
- `tests/test_train.py`: 9 CPU-only tests covering config, LoRA instantiation, LR scheduler
- `tests/test_export.py`: 14 CPU-only tests covering ROUGE-L, perplexity, mocked merge, manifest
- `docs/architecture.md`: Mermaid diagrams for pipeline, LoRA math, QLoRA memory, grad checkpointing
- `README.md`: Quick-start, benchmark table, config reference, format docs

[Unreleased]: https://github.com/avuppal/llm-finetuning/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/avuppal/llm-finetuning/releases/tag/v0.1.0
