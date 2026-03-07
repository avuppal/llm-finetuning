# Architecture — LLM Fine-tuning Pipeline

This document describes the system architecture for the `llm-finetuning` reference implementation.
All diagrams use [Mermaid](https://mermaid.js.org/) syntax (rendered on GitHub automatically).

---

## 1. Full Fine-tuning Pipeline

```mermaid
flowchart LR
    subgraph DATA ["📦 Data Preparation"]
        A[Raw Dataset\nAlpaca / ShareGPT] --> B[Prompt Formatter\ndataset.py]
        B --> C[Tokenizer\nAutoTokenizer]
        C --> D[Loss Masking\nlabels = -100 on prompt]
        D --> E[DataLoader\nbatched + padded]
    end

    subgraph TRAIN ["🏋️ Training"]
        F[Base Model\n4-bit NF4 weights] --> G[LoRA Injection\nPEFT get_peft_model]
        G --> H[Forward Pass\n+ CE Loss]
        H --> I[Backprop\ngradient checkpointing]
        I --> J[Paged AdamW\noptimizer step]
        J --> K[LR Scheduler\ncosine with warmup]
        K -->|next step| H
    end

    subgraph EVAL ["📊 Evaluation"]
        L[Held-out Split] --> M[Perplexity\nexp avg NLL]
        L --> N[Generation\n+ ROUGE-L]
    end

    subgraph EXPORT ["🚀 Export"]
        O[merge_and_unload\nW_new = W + AB·α/r]
        O --> P[save_pretrained\nsafetensors shards]
        P --> Q[Verify\nsmoke-test prompt]
    end

    E --> TRAIN
    K -->|checkpoint| EVAL
    EVAL -->|done| EXPORT
```

---

## 2. LoRA Mathematics

LoRA (**Lo**w-**R**ank **A**daptation) freezes the original weight matrix W and learns a
low-rank decomposition of the update ΔW.

```mermaid
graph TD
    subgraph "Standard Linear Layer  (frozen)"
        X["Input x\n∈ ℝᵈ"] --> W["W\n∈ ℝᵈˣᵏ  (frozen, 4-bit)"]
        W --> Y1["Wx\n∈ ℝᵏ"]
    end

    subgraph "LoRA Branch  (trainable)"
        X --> A["A  ∈ ℝʳˣᵈ\nr << d\n(init: random Gaussian)"]
        A --> B["B  ∈ ℝᵏˣʳ\n(init: zeros)"]
        B --> AB["BAx · (α/r)\n∈ ℝᵏ"]
    end

    Y1 --> Plus["+"]
    AB --> Plus
    Plus --> Y2["Output y = Wx + BAx·(α/r)\n∈ ℝᵏ"]

    style W fill:#f0f0f0,stroke:#999
    style A fill:#d4edda,stroke:#28a745
    style B fill:#d4edda,stroke:#28a745
```

**Key insight:** A ∈ ℝ^(r×d) and B ∈ ℝ^(k×r) together have only `r(d+k)` parameters
vs `dk` for the full weight matrix. With r=16, d=4096, k=4096: **131K vs 16.8M** params.

At initialisation B=0, so ΔW = BA = 0 and the adapter starts as an identity transform.

**Scaling:** The scaling factor α/r controls the effective learning rate of the adapter.
A larger α relative to r means the adapter updates are scaled up.

---

## 3. QLoRA Memory Breakdown

QLoRA combines 4-bit NF4 quantisation of the frozen base model with 16-bit LoRA adapters.

```mermaid
block-beta
    columns 3

    block:GPU["GPU VRAM — LLaMA-3 8B QLoRA (≈12 GB)"]
        A["🔵 4-bit NF4 Weights\n~4 GB\n(8B params × 0.5 bytes)"]
        B["🟢 LoRA Adapters A, B\n~64 MB\n(bfloat16, r=16)"]
        C["🟡 Quantisation Constants\n~200 MB\n(double quant: 8-bit scales)"]
        D["🟠 Activations\n~2–3 GB\n(with grad checkpointing)"]
        E["🔴 Optimizer States\n~400 MB\n(paged AdamW, 8-bit)"]
        F["🟣 KV-cache\ndisabled during training"]
    end
```

**Without QLoRA** (full bfloat16 fine-tuning):
- 8B × 2 bytes = **16 GB weights** + 16 GB gradients + 64 GB Adam states = **~96 GB**

**With QLoRA:**
- 8B × 0.5 bytes = **4 GB weights** (frozen, no gradients)
- Only LoRA params need gradients + optimizer states → **12 GB total**

---

## 4. Gradient Checkpointing — Memory / Compute Trade-off

```mermaid
graph TD
    subgraph "Standard Backprop  (more memory)"
        direction LR
        F1[Layer 1\nActivation ✓] --> F2[Layer 2\nActivation ✓]
        F2 --> F3[Layer 3\nActivation ✓]
        F3 --> F4[Layer N\nActivation ✓]
        F4 --> L1[Loss]
        L1 -->|"grad uses cached activations"| F3
        F3 -->|"grad uses cached activations"| F2
    end

    subgraph "Gradient Checkpointing  (less memory, +~33% compute)"
        direction LR
        G1[Layer 1\n✗ discarded] --> G2[Layer 2\n✗ discarded]
        G2 --> G3[Layer 3\n✗ discarded]
        G3 --> G4[Layer N\n✓ kept]
        G4 --> L2[Loss]
        L2 -->|"recompute fwd from checkpoint"| G3
        G3 -->|"recompute fwd from checkpoint"| G2
    end

    style F1 fill:#ffd,stroke:#cc0
    style F2 fill:#ffd,stroke:#cc0
    style F3 fill:#ffd,stroke:#cc0
    style F4 fill:#ffd,stroke:#cc0
    style G1 fill:#f0f0f0,stroke:#999
    style G2 fill:#f0f0f0,stroke:#999
    style G3 fill:#f0f0f0,stroke:#999
    style G4 fill:#ffd,stroke:#cc0
```

**Trade-off table:**

| Setting | Activation Memory | Compute Overhead | Typical Use |
|---------|------------------|------------------|-------------|
| No checkpointing | O(N) layers | None | Inference |
| Checkpoint every layer | O(1) | ~33% extra FLOPs | QLoRA training |
| Checkpoint every k layers | O(k) | ~1/k extra FLOPs | Full FT |

With gradient checkpointing:
- Activations are discarded after the forward pass
- During backprop, they are **recomputed** from the nearest checkpoint
- Net result: swap ~2 GB of activation memory for ~33% extra forward compute

In the QLoRA regime (frozen base, only adapter gradients), the activation overhead
is small since LoRA only modifies a few layers.

---

## 5. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI as train.py CLI
    participant Config as TrainingConfig
    participant DS as dataset.py
    participant Model as Base Model (4-bit)
    participant PEFT as LoRA Adapters
    participant Optim as Paged AdamW
    participant Sched as LR Scheduler
    participant Eval as evaluate.py
    participant Export as export.py

    User->>CLI: python -m src.train --config qlora_7b.yaml
    CLI->>Config: TrainingConfig.from_yaml()
    Config-->>CLI: cfg

    CLI->>DS: FineTuningDataset.from_config(cfg)
    DS-->>CLI: train_ds, val_ds

    CLI->>Model: AutoModelForCausalLM.from_pretrained(4-bit NF4)
    CLI->>PEFT: get_peft_model(model, LoraConfig)
    PEFT-->>CLI: lora_model (only A,B trainable)

    CLI->>Optim: PagedAdamW(lora_model.parameters())
    CLI->>Sched: cosine+warmup scheduler

    loop Training steps
        CLI->>Model: forward(batch) → loss
        CLI->>CLI: loss.backward()
        CLI->>Optim: optimizer.step()
        CLI->>Sched: scheduler.step()
    end

    CLI->>Eval: compute_perplexity(lora_model, val_loader)
    Eval-->>CLI: ppl=8.2

    CLI->>Export: merge_and_unload() + save_pretrained()
    Export-->>User: merged_model/ (safetensors)
```

---

## File Map

```
llm-finetuning/
├── src/
│   ├── dataset.py    ← prompt formatting, tokenisation, loss masking
│   ├── train.py      ← QLoRA training loop, config, optimiser
│   ├── evaluate.py   ← perplexity, ROUGE-L, generation
│   └── export.py     ← merge_and_unload, safetensors, manifest
├── configs/
│   ├── qlora_7b.yaml ← production 7-8B config
│   └── lora_1b.yaml  ← fast iteration / CI config
└── tests/            ← CPU-only pytest suite (no model downloads)
```
