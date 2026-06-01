# LLM From Scratch

A hands-on implementation of transformer-based language models built from the ground up in PyTorch. The project covers pretrained weight loading, instruction fine-tuning, LoRA adaptation, training diagnostics, and a Gradio-based chat frontend — all wired through a modular, multi-model architecture designed to scale as new models are added.

> Built while working through [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka, extended with Gemma3 support, LoRA fine-tuning, Apple Silicon (MPS) optimisation, and a multi-model frontend.

---

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training Diagnostics](#training-diagnostics)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Frontend Demo](#frontend-demo)
- [Extending with a New Model](#extending-with-a-new-model)
- [Roadmap](#roadmap)
- [References](#references)

---

## Features

- **Built from scratch** — attention, normalization, feedforward, and transformer blocks implemented directly in PyTorch
- **Multi-architecture** — GPT-2 (all sizes) and Gemma3-1B with a shared training pipeline; no model-specific branching in the trainer
- **Pretrained weight loading** — GPT-2 weights from OpenAI; Gemma3 weights from HuggingFace Hub with local disk caching (downloaded once, reused every run)
- **LoRA adaptation** — parameter-efficient fine-tuning via low-rank adapter matrices; toggled with a single CLI flag
- **Instruction fine-tuning** — Alpaca-style dataset pipeline with padding, target masking, and a custom collator
- **Training diagnostics** — 2×2 plot: loss curves, learning rate schedule, peak memory usage, and gradient norm
- **LR scheduling** — linear warmup followed by cosine decay; configurable via CLI
- **Best-model checkpointing** — weights saved automatically whenever validation loss improves
- **Device-aware** — automatic detection of CUDA, Apple Silicon (MPS), and CPU; MPS memory cache flushed after each optimizer step
- **Gradio frontend** — local web UI with model selector, temperature, and top-k controls
- **Modular registry** — adding a new model family requires changes to exactly two files

---

## Supported Models

| Model | Parameters | Weights Source | Status |
|---|---|---|---|
| GPT-2 Small | 124M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 Medium | 355M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 Large | 774M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 XL | 1558M | OpenAI (auto-downloaded) | ✅ Inference only on 16 GB MPS (OOM during training) |
| Gemma3-1B | 1B | HuggingFace Hub (gated) | ⚠️ Implemented, not yet tested end-to-end |

> **Gemma3-1B note:** The architecture, weights loader, and key mapping are implemented and code-complete. End-to-end training has not yet been verified — treat results as unvalidated until a test run is confirmed.

---

## Project Structure

```
llmFromScratch/
├── src/
│   ├── main.py                        # Training entry point (CLI)
│   ├── models/
│   │   ├── base_model.py              # BaseLanguageModel ABC
│   │   ├── registry.py                # Model + loader factory, weight/metric persistence
│   │   ├── gpt2/                      # GPT-2 architecture
│   │   │   ├── attention.py           # Multi-head causal self-attention
│   │   │   ├── feedforward.py         # GELU feedforward block
│   │   │   ├── normalization.py       # LayerNorm
│   │   │   ├── transformer.py         # TransformerBlock
│   │   │   └── gpt.py                 # GPTModel (full architecture)
│   │   └── gemma3/                    # Gemma3-1B architecture
│   │       ├── attention.py           # Grouped Query Attention + RoPE + QK-norm
│   │       ├── feedforward.py         # SwiGLU feedforward block
│   │       ├── normalization.py       # RMSNorm
│   │       ├── rope_wrapper.py        # RoPE frequency computation and application
│   │       ├── transformer.py         # TransformerBlock (sliding/full attention)
│   │       └── gemma.py               # GemmaModel (full architecture)
│   ├── training/
│   │   ├── trainer.py                 # ModelTrainer — training loop, evaluation, generation
│   │   ├── loss.py                    # LossCalibrator — batch and loader loss helpers
│   │   └── dataloader.py
│   ├── data/
│   │   ├── instruction_dataset.py     # InstructionDataset — Alpaca-format tokenisation
│   │   └── instruction_collator.py    # InstructionCollator — padding + target masking
│   └── utils/
│       ├── config.py                  # Hyperparameter configs for all models
│       ├── lora.py                    # LoRA — apply_lora, enable_lora, disable_lora
│       ├── plotting.py                # 2×2 training diagnostics figure
│       ├── tokenizer_adapter.py       # HFTokenizerAdapter — tiktoken-compatible wrapper
│       ├── model_inference.py         # InferenceEngine
│       ├── gpt_download.py            # GPT-2 checkpoint downloader
│       └── weights_loader/
│           ├── base_weights_loader.py # BaseWeightsLoader ABC
│           ├── hf_weights_loader.py   # Abstract HuggingFace loader (download + cache)
│           ├── gpt2_weights_loader.py # GPT-2 key mapping
│           └── gemma3_weights_loader.py # Gemma3 key mapping
├── frontend/
│   └── app.py                         # Gradio multi-model chat UI
├── documentations/
│   └── design_choice.md               # Architecture and design decisions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

**Prerequisites:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/<your-username>/llmFromScratch.git
cd llmFromScratch

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

GPT-2 pretrained weights and the instruction dataset are downloaded automatically on first run.

### Gemma3-1B additional setup

Gemma3 is a gated model — complete these steps once before running:

1. Accept the license at [huggingface.co/google/gemma-3-1b](https://huggingface.co/google/gemma-3-1b)
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Run `huggingface-cli login` and paste your token when prompted

Weights download to `data/gemma3_cache/` on first run and are reused on every subsequent run (no network call).

---

## Usage

### Fine-tune a model

```bash
# Fine-tune GPT-2 Small with defaults
python -m src.main

# Fine-tune Gemma3-1B
python -m src.main --model gemma3-1b

# Fine-tune GPT-2 Medium with custom settings
python -m src.main --model "gpt2-medium (355M)" --epochs 3 --lr 2e-4 --warmup-steps 200
```

Fine-tuned weights are saved to `data/fine_tuned_weights/<model-name>.pth`. The best checkpoint (lowest validation loss) is also saved during training — not just at the end.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2-small (124M)` | Model to train |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `0.0004` | Peak learning rate |
| `--warmup-steps` | `100` | Linear LR warmup steps |
| `--batch-size` | model default | Override micro-batch size |
| `--max-length` | model default | Override max sequence length |
| `--grad-accum` | model default | Override gradient accumulation steps |
| `--lora` | off | Enable LoRA fine-tuning |
| `--lora-rank` | `8` | LoRA rank `r` |
| `--lora-alpha` | `16.0` | LoRA scaling factor |

### Run the frontend

```bash
python -m frontend.app
```

Opens a Gradio UI at `http://127.0.0.1:7860`.

---

## Training Diagnostics

After each training run, a metrics JSON is saved alongside the weights. The `plot_loss_curves` utility in `src/utils/plotting.py` reads this file and produces a 2×2 diagnostic figure:

| Panel | What it shows |
|---|---|
| **Train / Val Loss** | Cross-entropy loss vs tokens seen |
| **Learning Rate** | Warmup ramp + cosine decay curve |
| **Peak Memory (GB)** | GPU/MPS memory usage at each eval step |
| **Gradient Norm** | Pre-clip L2 norm with clip ceiling marked |

The gradient norm panel makes it immediately obvious whether clipping is always active (norm == 0.5 throughout indicates the LR or initialisation may need tuning).

---

## LoRA Fine-tuning

LoRA freezes all pretrained weights and adds small trainable adapter matrices (`A` and `B`) to the target attention projections. Only the adapter parameters are updated during training.

```bash
# LoRA fine-tuning with defaults (r=8, alpha=16)
python -m src.main --model gemma3-1b --lora

# Higher rank for more capacity
python -m src.main --model gemma3-1b --lora --lora-rank 16 --lora-alpha 32
```

**Memory benefit:** AdamW allocates moment buffers only for trainable parameters. With LoRA active on a 1B parameter model, optimizer state drops from ~8 GB to a few MB. Combined with gradient accumulation this makes Gemma3-1B trainable on 16 GB MPS.

**Default target layers:** `W_query` and `W_value` in every attention block (works identically for GPT-2 and Gemma3 — both use the same attribute names).

To compare base vs adapted outputs at inference without reloading weights:

```python
from src.utils.lora import disable_lora, enable_lora
disable_lora(model)   # behaves as frozen pretrained checkpoint
enable_lora(model)    # adapter delta re-applied
```

---

## Frontend Demo

The Gradio UI (`frontend/app.py`) supports:

- Model selection from all trained/available checkpoints
- Adjustable max token length (50–512)
- Adjustable temperature (0.1–1.5)
- Automatic fallback to pretrained weights if no fine-tuned checkpoint exists

```bash
python -m frontend.app
```

---

## Extending with a New Model

### From HuggingFace Hub (recommended)

Subclass `HFWeightsLoader` and implement `_map_keys()` to translate HuggingFace key names to this codebase's naming:

```python
# src/utils/weights_loader/mymodel_weights_loader.py
from src.utils.weights_loader.hf_weights_loader import HFWeightsLoader

class MyModelWeightsLoader(HFWeightsLoader):
    def __init__(self):
        super().__init__("org/model-name", cache_dir="data/mymodel_cache")

    def _map_keys(self, hf_state_dict, model):
        mapped = {}
        mapped["tok_emb.weight"] = hf_state_dict["model.embed_tokens.weight"]
        # ... rest of the key mapping ...
        return mapped
```

Download, caching, shard merging, and `load_state_dict` are handled by `HFWeightsLoader` automatically.

### Register the model

Add two lines to `src/models/registry.py`:

```python
_MODELS["my-model"] = MyModel
_LOADERS["my-model"] = lambda: MyModelWeightsLoader()
```

And one entry to `src/utils/config.py`:

```python
"my-model": {
    "vocab_size": ...,
    "context_length": ...,
    ...
}
```

`main.py`, the trainer, and the frontend all work without any further changes.

---

## Roadmap

- [x] GPT-2 architecture from scratch (all four sizes)
- [x] Pretrained weight loading — GPT-2 (OpenAI) and Gemma3-1B (HuggingFace Hub)
- [x] Instruction fine-tuning pipeline (Alpaca format)
- [x] Multi-model registry
- [x] Gradio multi-model frontend
- [x] Apple Silicon (MPS) support with memory management
- [x] Gradient accumulation, clipping, and AdamW tuning
- [x] LR warmup + cosine decay scheduler
- [x] Best-model checkpointing during training
- [x] LoRA parameter-efficient fine-tuning
- [x] Training diagnostic plots (loss, LR, memory, grad norm)
- [x] HuggingFace weights loader with disk caching
- [ ] Evaluation metrics (BLEU, ROUGE)
- [ ] HuggingFace Spaces deployment

---

## References

- Raschka, S. — [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. — [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Google — [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)
- Hu et al. — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Stanford Alpaca instruction dataset](https://github.com/tatsu-lab/stanford_alpaca)
