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
- [Google Colab](#google-colab)
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
- **Training diagnostics** — 2×3 diagnostic dashboard: loss, overfitting gap, LR schedule, gradient norm, memory usage, and step throughput — with self-diagnosing warnings on each panel
- **LR scheduling** — linear warmup followed by cosine decay; configurable via CLI
- **Best-model checkpointing** — weights saved automatically whenever validation loss improves
- **Device-aware** — automatic detection of CUDA, Apple Silicon (MPS), and CPU; MPS memory cache flushed after each optimizer step
- **Gradio frontend** — multi-tab web UI: Generate (chat), Compare Models (LoRA vs Full comparison table and per-model loss curves), Loss Curves (per-checkpoint diagnostic dashboard)
- **LoRA checkpoint efficiency** — adapter-only saves (`_lora.pth` stores only adapter matrices, not the full model); config sidecar (`_lora_config.json`) records rank, alpha, and target modules for exact reproduction
- **Post-training evaluation** — BERTScore F1 (semantic similarity) and ROUGE-L scored on the held-out test split; results saved to the metrics JSON and surfaced in all comparison plots
- **Modular registry** — adding a new model family requires changes to exactly two files

---

## Supported Models

| Model | Parameters | Weights Source | Status |
|---|---|---|---|
| GPT-2 Small | 124M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 Medium | 355M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 Large | 774M | OpenAI (auto-downloaded) | ✅ Tested — full fine-tuning + LoRA |
| GPT-2 XL | 1558M | OpenAI (auto-downloaded) | ✅ Inference only on 16 GB MPS (OOM during training) |
| Gemma3-1B | 1B | HuggingFace Hub (gated) | ✅ Tested — full fine-tuning + LoRA |

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
│   ├── evaluation/
│   │   └── metrics.py                 # BERTScore F1 + ROUGE — evaluate(predictions, references)
│   └── utils/
│       ├── config.py                  # Hyperparameter configs for all models
│       ├── lora.py                    # LoRA — apply_lora, enable_lora, disable_lora
│       ├── plotting.py                # Training diagnostics dashboard + comparison plots
│       ├── tokenizer_adapter.py       # HFTokenizerAdapter — tiktoken-compatible wrapper
│       ├── model_inference.py         # InferenceEngine
│       ├── gpt_download.py            # GPT-2 checkpoint downloader
│       └── weights_loader/
│           ├── base_weights_loader.py # BaseWeightsLoader ABC
│           ├── hf_weights_loader.py   # Abstract HuggingFace loader (download + cache)
│           ├── gpt2_weights_loader.py # GPT-2 key mapping
│           └── gemma3_weights_loader.py # Gemma3 key mapping
├── notebooks/
│   └── colab_runner.ipynb             # One-click Colab launcher (Drive mount, clone, train, plot)
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

1. Accept the license at [huggingface.co/google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt)
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
| `--grad-clip` | `0.5` | Max gradient L2 norm for clipping |
| `--eval-freq` | `20` | Evaluate every N optimizer steps |
| `--eval-iter` | `30` | Batches used per evaluation |
| `--eval-samples` | `200` | Test entries scored with BERTScore/ROUGE after training (`0` to skip) |
| `--bertscore-model` | `distilbert-base-uncased` | HuggingFace model used by BERTScore |

### Run the frontend

```bash
python -m frontend.app
```

Opens a Gradio UI at `http://127.0.0.1:7860`.

---

## Training Diagnostics

After each training run, a metrics JSON is saved alongside the weights. The `plot_loss_curves` utility in `src/utils/plotting.py` reads this file and produces a **2×3 dashboard** with self-diagnosing warning annotations on each panel:

| Position | Panel | What it shows |
|---|---|---|
| Top-left | **Train / Val Loss** | Loss curves with epoch boundary markers |
| Top-centre | **Overfitting Gap** | Val − Train loss gap; shaded when positive; warns if widening late |
| Top-right | **Learning Rate** | Warmup ramp + cosine decay; warns if schedule didn't fire |
| Bottom-left | **Gradient Norm** | Pre-clip L2 norm with clip-ceiling line and shaded clipped region |
| Bottom-centre | **Peak Memory** | GPU/MPS memory per eval step; shows GPU utilisation % on CUDA |
| Bottom-right | **Step Throughput** | Per-step time series (or tok/s summary card if step times unavailable) |

Each panel flags actionable problems automatically — constant clipping, widening overfitting gap, GPU under-utilisation, and low throughput — with a suggested fix inline.

The run summary (effective batch size, peak LR, epoch count, LoRA r/α when applicable, training time, tok/s) is displayed as a subtitle under the figure title. When BERTScore/ROUGE evaluation was run, scores appear as a third subtitle line. Spike filtering removes timing outliers from the step-time panel automatically.

---

## Evaluation Metrics

After training, `main.py` generates responses for up to `--eval-samples` entries from the **held-out test split** and scores them against the ground-truth references using two metrics:

| Metric | What it measures | Why not BLEU |
|---|---|---|
| **BERTScore F1** | Semantic similarity via contextual embeddings — paraphrases score high | BLEU penalises valid rewording |
| **ROUGE-L** | Longest common subsequence overlap — lightweight relative comparison | Kept as a cheap complement |

```bash
# Default: score 200 test entries after training
python -m src.main --model "gpt2-small (124M)"

# Faster scorer model, fewer samples
python -m src.main --model "gpt2-small (124M)" --eval-samples 50

# Skip evaluation entirely
python -m src.main --model "gpt2-small (124M)" --eval-samples 0
```

Scores are saved to the metrics JSON (`bertscore_f1`, `rougeL`, etc.) and automatically appear in all three plot types — the loss curve subtitle, the per-model comparison table, and the All master table.

Both `bert-score` and `rouge-score` are **optional** — if either package is missing, that scorer is skipped with a warning and the other still runs. Training is never blocked.

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

**Adapter-only checkpoints:** `_lora.pth` stores only the trained adapter matrices (tens of MB), not the frozen base weights. A `_lora_config.json` sidecar records rank, alpha, and target modules so the adapter can be re-applied exactly at inference. Old full-state-dict checkpoints are still loadable without migration.

**Default target layers:** `W_query` and `W_value` in every attention block (works identically for GPT-2 and Gemma3 — both use the same attribute names).

To compare base vs adapted outputs at inference without reloading weights:

```python
from src.utils.lora import disable_lora, enable_lora
disable_lora(model)   # behaves as frozen pretrained checkpoint
enable_lora(model)    # adapter delta re-applied
```

---

## Frontend Demo

The Gradio UI (`frontend/app.py`) has three tabs:

**Generate** — chat with any trained checkpoint
- Model selector (auto-discovers all available full and LoRA checkpoints)
- Adjustable max token length (50–512) and temperature (0.1–1.5)
- Automatic fallback to pretrained weights if no fine-tuned checkpoint exists
- Input format: plain instruction text — the Alpaca prompt wrapper is added automatically

**Compare Models** — LoRA vs Full comparison across all trained models
- `All` view: master comparison table (Best Val↓, Perplexity↓, Best Step, Time, Tok/s, Peak Mem, GPU Util) with winner cells highlighted
- Per-model view: side-by-side LoRA vs Full loss curves + per-model comparison table with the same metrics
- Subtitles show shared training config (lr, epochs, bs) and LoRA rank/α when applicable
- Rendered automatically on page load — no button click required

**Loss Curves** — 2×3 diagnostic dashboard per checkpoint (see [Training Diagnostics](#training-diagnostics))

```bash
python -m frontend.app
```

---

## Google Colab

The project runs on Google Colab via the **Google Colab** VS Code extension or directly at [colab.research.google.com](https://colab.research.google.com).

Open `notebooks/colab_runner.ipynb` and run the cells in order:

| Cell | What it does |
|------|-------------|
| 1 | Mounts Google Drive — weights and checkpoints are saved here and survive runtime restarts |
| 2 | Clones the repo on first run, `git pull` on subsequent runs, clears `__pycache__` |
| 3 | Installs dependencies |
| 4 | HuggingFace login (Gemma3 only — skip for GPT-2) |
| 5 | Trains the model (edit flags as needed) |
| 6 | Optional LoRA fine-tuning on Gemma3-1B |
| 7 | Plots training diagnostics and saves the figure |

**Workflow:** edit code locally in VS Code → `git push` → re-run Cell 2 in Colab to pull latest → train.

**Persistence:** all weights, checkpoints, and HuggingFace model cache are saved to `MyDrive/llmFromScratch/` automatically — no manual copying needed.

> **Important:** always push your latest code to GitHub before running Colab. The runtime clones directly from GitHub; local files are not accessible there.

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
- [x] Gradio multi-model frontend — Generate, Compare Models (LoRA vs Full), and Loss Curves tabs
- [x] Apple Silicon (MPS) support with memory management
- [x] Gradient accumulation, clipping, and AdamW tuning
- [x] LR warmup + cosine decay scheduler
- [x] Best-model checkpointing during training
- [x] LoRA parameter-efficient fine-tuning
- [x] Training diagnostic dashboard — 2×3 plots with self-diagnosing warnings
- [x] HuggingFace weights loader with disk caching
- [x] Post-training evaluation — BERTScore F1 + ROUGE-L on held-out test split
- [ ] HuggingFace Spaces deployment

---

## References

- Raschka, S. — [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. — [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Google — [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)
- Hu et al. — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Stanford Alpaca instruction dataset](https://github.com/tatsu-lab/stanford_alpaca)
