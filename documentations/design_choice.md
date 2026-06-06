# Architecture & Design Choices

This document captures the key architectural decisions and design patterns used in this project. It is intended as a reference for contributors and reviewers.

---

## Project Goal

Build a modular, from-scratch LLM training framework in PyTorch that supports multiple model architectures (starting with GPT-2 and Gemma3) using a shared training pipeline. The framework is designed for instruction fine-tuning, with the ability to load official pretrained weights from HuggingFace as a starting point.

---

## Repository Layout

```
src/
├── main.py                        # CLI entry point
├── models/
│   ├── base_model.py              # Abstract base class for all models
│   ├── registry.py                # Model + loader lookup, weight/metric persistence
│   ├── gpt2/                      # GPT-2 implementation
│   │   ├── attention.py           # MultiHeadAttention
│   │   ├── feedforward.py
│   │   ├── normalization.py
│   │   ├── transformer.py
│   │   └── gpt.py                 # GPTModel (top-level)
│   └── gemma3/                    # Gemma3 implementation
│       ├── attention.py           # GroupedQueryAttention
│       ├── feedforward.py         # SwiGLU FFN
│       ├── normalization.py       # RMSNorm
│       ├── rope_wrapper.py        # RoPE utilities
│       ├── transformer.py         # TransformerBlock (sliding/full attention)
│       └── gemma.py               # GemmaModel (top-level)
├── training/
│   ├── trainer.py                 # ModelTrainer (shared across all models)
│   ├── loss.py                    # LossCalibrator
│   ├── dataloader.py
│   ├── instruction_dataset.py
│   └── instruction_collator.py
└── utils/
    ├── config.py                  # Model hyperparameter configs
    ├── tokenizer_adapter.py       # HFTokenizerAdapter (tiktoken-compatible wrapper)
    ├── model_inference.py
    └── weights_loader/
        ├── base_weights_loader.py # Abstract loader interface
        ├── hf_weights_loader.py   # Abstract HuggingFace loader (download + cache)
        ├── gpt2_weights_loader.py # GPT-2 specific key mapping
        └── gemma3_weights_loader.py # Gemma3 specific key mapping
```

---

## Core Design Patterns

### 1. Abstract Base Classes for Model and Loader

Every model inherits from `BaseLanguageModel` (`src/models/base_model.py`), which enforces three contracts:

```python
class BaseLanguageModel(nn.Module, ABC):
    def forward(self, x): ...
    def get_context_size(self) -> int: ...
    def get_tokenizer(self): ...
```

This allows the trainer, inference utilities, and registry to work with any model without knowing its internals — swapping GPT-2 for Gemma3 requires only changing the `--model` CLI flag.

Every weights loader inherits from `BaseWeightsLoader`, which enforces a single `__call__(model)` interface. Loading weights into any model is always one line: `loader(model)`.

### 2. Three-Level Weights Loader Hierarchy

```
BaseWeightsLoader          (abstract __call__ interface)
└── HFWeightsLoader        (abstract; handles HuggingFace download + caching)
    └── Gemma3WeightsLoader (concrete; implements _map_keys() for Gemma3)
```

`HFWeightsLoader` handles all the infrastructure that is the same for every HuggingFace model: downloading via `snapshot_download`, merging weight shards (`.safetensors` or `.bin`), and calling `load_state_dict`. The only thing a new model needs to provide is `_map_keys(hf_state_dict, model) -> dict` — a translation table from HuggingFace key names to this codebase's key names.

**To add a new HuggingFace model:**
```python
class MyModelWeightsLoader(HFWeightsLoader):
    def __init__(self):
        super().__init__("org/model-name", cache_dir="data/mymodel_cache")

    def _map_keys(self, hf_state_dict, model):
        mapped = {}
        mapped["tok_emb.weight"] = hf_state_dict["model.embed_tokens.weight"]
        # ... rest of the key mapping ...
        return mapped
```

### 3. Cache-First Weight Loading

`HFWeightsLoader._download()` always tries `local_files_only=True` first. If the weights are already on disk, no network call is made at all. On a cache miss, it falls back to a full download and then populates the cache. This means the first run downloads once; every subsequent run is instant.

```
Run 1: snapshot_download(local_files_only=True) → cache miss
        snapshot_download() → downloads to data/gemma3_cache/
Run 2+: snapshot_download(local_files_only=True) → returns immediately from disk
```

### 4. Model Registry

`src/models/registry.py` is the single source of truth for which class and which loader correspond to each model name. Adding a model means adding two lines — one in `_MODELS` and one in `_LOADERS`.

```python
_MODELS = {
    "gpt2-small (124M)":  GPTModel,
    "gemma3-1b":          GemmaModel,
    ...
}
_LOADERS = {
    "gpt2-small (124M)":  lambda: GPT2WeightsLoader("124M"),
    "gemma3-1b":          lambda: Gemma3WeightsLoader(),
    ...
}
```

### 5. Tokenizer Abstraction

GPT-2 uses `tiktoken`; Gemma3 uses a HuggingFace `AutoTokenizer`. Rather than branching throughout the codebase, each model's `get_tokenizer()` returns an object that exposes a tiktoken-compatible interface (`encode`, `decode`, `eot_token`). For HuggingFace tokenizers, `HFTokenizerAdapter` (`src/utils/tokenizer_adapter.py`) wraps the HF tokenizer to match that interface. The trainer and inference code never need to know which tokenizer is in use.

---

## Model Architectures

### GPT-2

| Component | Implementation |
|-----------|---------------|
| Attention | Multi-head self-attention with causal mask (learnable QKV projections) |
| Positional encoding | Learned absolute embeddings |
| Normalization | LayerNorm (pre-norm applied before attention and FFN) |
| Feed-forward | Two-layer MLP with GELU activation |
| Weights source | OpenAI's official GPT-2 checkpoints (downloaded via `gpt_download.py`) |

### Gemma3-1B

> **Status:** Architecture and weights loader are implemented and code-complete. End-to-end training has not yet been verified on hardware — treat as unvalidated until a test run is confirmed.

Matches the `google/gemma-3-1b-pt` architecture exactly so pretrained HuggingFace weights transfer without reshaping.

| Component | Implementation |
|-----------|---------------|
| Attention | Grouped Query Attention (GQA) with 4 query heads and 1 KV group |
| Positional encoding | Rotary Position Embeddings (RoPE) — no learned pos embeddings |
| Normalization | RMSNorm with `(1 + weight)` scaling pattern |
| Feed-forward | SwiGLU (gated linear unit with Swish activation, 3 weight matrices) |
| Attention pattern | Alternating: 5 sliding-window layers → 1 full-attention layer, repeated 4 times + 2 trailing sliding (26 layers total) |
| QK normalization | Per-head RMSNorm on queries and keys before attention scores |
| Sliding window | Local attention window of 512 tokens; full-attention layers see the entire context |
| Weight tying | `lm_head.weight` is tied to `embed_tokens.weight` (Gemma3 uses the same matrix for input and output) |
| Weights source | `google/gemma-3-1b-pt` on HuggingFace Hub (gated — requires license acceptance) |

#### RoPE: Local vs Global

Two separate sets of RoPE frequency tables are precomputed and registered as non-persistent buffers:

- **Local RoPE** (`rope_local_base=10_000`): used by sliding-window attention layers
- **Global RoPE** (`rope_base=1_000_000`): used by full-attention layers

This matches the Gemma3 paper's design, where local layers use a shorter-range frequency basis appropriate for the 512-token window.

#### Sliding Window Mask Construction

Masks are constructed once per forward pass in `GemmaModel._create_masks()`:

- `mask_global`: standard upper-triangular causal mask (future tokens masked)
- `far_past`: masks positions more than `sliding_window` tokens in the past
- `mask_local = mask_global | far_past`: each sliding-window layer sees only a 512-token window

---

## Training Pipeline

### Shared Trainer (`ModelTrainer`)

The same `ModelTrainer` class is used for both GPT-2 and Gemma3 with no model-specific branching. It accepts the model, data loaders, optimizer, and optional scheduler/model-name at call time.

### Gradient Accumulation

Gradient accumulation was introduced specifically to address out-of-memory (OOM) errors encountered during fine-tuning on Apple M4 with 16 GB unified memory. Without it, even GPT-2 Medium at `batch_size=4` exceeded available memory. With accumulation, a micro-batch of 1 is run multiple times before each optimizer step, achieving an equivalent large-batch update at a fraction of the peak memory cost.

The loss is divided by `gradient_accumulation_steps` before `.backward()` so that accumulated gradients equal a single full-batch gradient in magnitude. The optimizer steps only after the full accumulation cycle completes.

```
effective_batch_size = batch_size × gradient_accumulation_steps
```

Per-model defaults in `src/utils/config.py` are tuned for 16 GB Apple Silicon MPS:

| Model | batch_size | grad_accum | effective_batch |
|-------|-----------|------------|----------------|
| GPT-2 Small | 8 | 1 | 8 |
| GPT-2 Medium | 1 | 4 | 4 |
| GPT-2 Large | 1 | 4 | 4 |
| Gemma3-1B | 1 | 8 | 8 |

All values can be overridden via `--batch-size` and `--grad-accum` CLI flags without touching the config file.

### Gradient Clipping

The L2 norm of all parameter gradients is clipped to `max_norm=0.5` before each optimizer step. If the combined gradient vector exceeds this norm, every gradient is scaled down proportionally. This prevents unstable large updates from a single bad batch — especially important when loading pretrained weights and fine-tuning on a new distribution.

### Optimizer

AdamW with transformer-standard hyperparameters:

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| `lr` | 0.0004 (default, CLI-overridable) | Starting point for fine-tuning |
| `weight_decay` | 0.1 | Standard L2 regularisation for transformers |
| `beta1` | 0.9 | Standard first moment |
| `beta2` | 0.95 | Faster decay of second moment vs PyTorch default 0.999; matches GPT-3 / Chinchilla |

### LR Scheduler: Warmup + Cosine Decay

A `SequentialLR` combines two phases, stepped per optimizer update (not per epoch):

1. **Linear warmup**: ramps from 1% of peak LR to peak LR over `warmup_steps` (default 100). Prevents large destabilising updates at the start of training when weights are not yet adapted.
2. **Cosine decay**: smoothly reduces LR from peak down to 10% of peak over the remaining steps. Avoids oscillation around the loss minimum that a fixed LR causes late in training.

### Best-Model Checkpointing

During training, whenever validation loss improves, weights are saved to `data/fine_tuned_weights/`. This guards against overfitting: if validation loss improves early then rises, the saved weights capture the best-generalising point rather than the end of training.

### MPS Memory Management

`torch.mps.empty_cache()` is called after each optimizer step and after each epoch to return fragmented memory to the OS pool. This prevents the progressive memory fragmentation that can cause OOM errors on Apple Silicon during long training runs.

### Step Timing

`ModelTrainer` records the wall-clock duration of every optimizer step in `self.step_times_sec`. The first step initialises the timer; each subsequent step appends the elapsed time since the previous update. After training, `main.py` saves this array alongside the other metrics and computes an aggregate `tokens_per_sec` figure for the run summary.

### Training Diagnostics Dashboard

`src/utils/plotting.py` reads the saved metrics JSON and renders a **2×3 figure** with six panels. Each panel includes a self-diagnosing annotation that detects common problems and suggests a concrete fix:

| Position | Panel | Diagnostic check |
|---|---|---|
| Top-left | Train / Val Loss | Warns if overfitting gap widens or val loss plateaus |
| Top-centre | Overfitting Gap (Val − Train) | Flags if late gap > 1.5× early gap |
| Top-right | Learning Rate Schedule | Warns if LR barely changed (scheduler not stepping) |
| Bottom-left | Gradient Norm (pre-clip) | Warns if clipped on >80% or >30% of steps |
| Bottom-centre | Peak Memory | Shows GPU utilisation %; warns if under 40% |
| Bottom-right | Step Throughput | Per-step time series or tok/s summary card; warns if < 200 tok/s |

A run-summary subtitle (effective batch, peak LR, epochs, training time, tok/s) is rendered under the figure title. Panels where data is absent show a "No data" placeholder — backward compatible with metric JSONs saved before new keys were added.

**Metrics saved per run:**

| Key | Source | Type |
|-----|--------|------|
| `train_losses`, `val_losses` | eval loop | list per eval step |
| `tokens_seen` | training loop | list per eval step |
| `learning_rates`, `grad_norms`, `peak_memory_gb` | eval loop | list per eval step |
| `step_times_sec` | optimizer step | list per optimizer step |
| `tokens_per_sec` | post-training | scalar (run average) |
| `grad_clip_norm` | constant | scalar (0.5) |
| `gpu_memory_total_gb` | device query | scalar or None (MPS/CPU) |

---

## Configuration

All model hyperparameters live in `src/utils/config.py` as a single dictionary. Each model entry includes both architectural dimensions and training memory defaults (`batch_size`, `allowed_max_length`, `gradient_accumulation_steps`). CLI flags (`--batch-size`, `--max-length`, `--grad-accum`, `--lr`, `--warmup-steps`) override these defaults without modifying the config file.

---

## Data Pipeline

The project fine-tunes on an instruction-following dataset (Alpaca-style JSON). The pipeline:

1. `InstructionDataset` formats each entry into `### Instruction / ### Input / ### Response` prompt structure and tokenises it
2. `InstructionCollator` pads sequences to a uniform length within each batch and sets prompt-token labels to `-100` so cross-entropy loss is computed only on the response tokens
3. An 85/10/5 train/test/val split is applied; test entries are used for sample inference after training

---

## LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that inserts small trainable adapter matrices alongside frozen pretrained weights. Instead of updating all model parameters, only the adapter matrices are trained — dramatically reducing memory and compute requirements for fine-tuning.

### How It Works

For a frozen weight matrix `W ∈ R^{d_out × d_in}`, LoRA adds a low-rank delta:

```
y = Wx + (alpha/r) * B A x
```

where `A ∈ R^{r × d_in}` and `B ∈ R^{d_out × r}` are the trainable adapters and `r << d_in`. `B` is initialized to zero so the adapter contributes nothing at the start of training — the model begins in an identical state to the pretrained checkpoint.

### Implementation

`src/utils/lora.py` provides three public functions:

| Function | Description |
|----------|-------------|
| `apply_lora(model, r, alpha, target_modules)` | Freezes all base weights, wraps target `nn.Linear` layers with `LoRALinear`. Call after loading pretrained weights, before creating the optimizer. |
| `enable_lora(model)` | Re-enables the LoRA delta in all adapters (active by default). |
| `disable_lora(model)` | Disables the LoRA delta so the model behaves as the frozen base checkpoint — useful for comparing base vs adapted behaviour at inference. |

`LoRALinear` is a drop-in replacement for `nn.Linear` that holds the original frozen weight alongside the trainable `lora_A` and `lora_B` parameters. It respects the base layer's dtype, so bfloat16 Gemma3 models get bfloat16 adapters automatically.

The adapter forward pass splits the chained matmul into two steps to avoid numerical overflow in bfloat16:
```python
lora_out = (x @ lora_A.T) @ lora_B.T   # two steps, not x @ lora_A.T @ lora_B.T
out = out + lora_out * scaling
```

### Target Modules

Both GPT-2 and Gemma3 use the same attribute names for their attention projections (`W_query`, `W_key`, `W_value`, `out_proj`), so the same `target_modules` tuple works for both models. The default targets `W_query` and `W_value` — the minimal configuration from the original LoRA paper. More projections can be added via the CLI.

### Memory Benefit

When LoRA is active, the optimizer only receives parameters with `requires_grad=True` (the adapter matrices). AdamW therefore allocates moment buffers only for those — not the frozen base weights. For a ~1B parameter model this avoids ~8 GB of optimizer state that would otherwise be needed for full fine-tuning.

### Usage

```bash
# Full fine-tuning (default — all parameters trained)
python -m src.main --model gemma3-1b

# LoRA fine-tuning (only adapter weights trained)
python -m src.main --model gemma3-1b --lora

# Custom rank and alpha
python -m src.main --model gemma3-1b --lora --lora-rank 16 --lora-alpha 32
```

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--lora` | off | Enable LoRA (flag, no value needed) |
| `--lora-rank` | 8 | Adapter rank `r`. Higher = more capacity, more parameters. |
| `--lora-alpha` | 16.0 | Scaling factor. Effective scale = `alpha / r`. |

---

## Google Colab and Google Drive

### Environment Detection (`src/utils/colab.py`)

`get_data_dir()` is called at the moment a path is needed (not at import time) and returns:

| Environment | Path returned |
|-------------|--------------|
| Local / non-Colab | `data/` (relative to working directory) |
| Colab — Drive mounted | `/content/drive/MyDrive/llmFromScratch/` |
| Colab — Drive not yet mounted | Mounts Drive automatically, then returns Drive path |
| Colab — Drive mount fails | Falls back to `/content/data/` with a warning |

All weight loaders (`GPT2WeightsLoader`, `Gemma3WeightsLoader`, `HFWeightsLoader`) and the model registry call `get_data_dir()` when constructing paths, so checkpoints, HuggingFace caches, and fine-tuned weights all land in Google Drive automatically when running on Colab.

### Colab Runner Notebook (`notebooks/colab_runner.ipynb`)

A single notebook that handles the full Colab setup: Drive mount → git clone/pull → dependency install → HuggingFace login → training → diagnostic plot.

**Why git clone is required even when using VS Code:** The Colab runtime runs on Google's servers. VS Code (via the Google Colab extension) is the UI only — local project files are not accessible in the runtime. Code must be pushed to GitHub first and pulled in Colab.

The notebook clears `__pycache__` after every clone/pull to prevent stale bytecode from shadowing updated source files — a common cause of "wrong repo ID" errors when the code is updated between sessions.

### Naming: Internal vs HuggingFace

| Identifier | Value | Where used |
|-----------|-------|-----------|
| Internal model name | `gemma3-1b` | CLI `--model`, config keys, registry keys |
| HuggingFace repo | `google/gemma-3-1b-pt` | `_GEMMA3_1B_REPO`, weights download URL |
| HuggingFace tokenizer | `google/gemma-3-1b-pt` | `tokenizer_id` in config, `AutoTokenizer.from_pretrained` |

These are kept separate deliberately — the internal name is stable across any future model renames on HuggingFace.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core deep learning framework |
| `tiktoken` | GPT-2 tokenizer |
| `huggingface_hub` | `snapshot_download` for HF model checkpoints |
| `safetensors` | Efficient weight shard loading (preferred over `.bin`) |
| `transformers` | `AutoTokenizer` for Gemma3 and other HF models |

Large binary artefacts (`data/gpt2/`, `data/gemma3_cache/`, `data/hf_cache/`, `data/fine_tuned_weights/`) are excluded from the repository via `.gitignore` and are downloaded or generated at runtime.

---

## Prerequisites for Gemma3

Gemma3 is a gated model on HuggingFace. Before running with `--model gemma3-1b`:

1. Accept the license at [huggingface.co/google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt)
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Run `huggingface-cli login` and paste your token when prompted

Weights are downloaded once to `data/gemma3_cache/` and reused on all subsequent runs.
