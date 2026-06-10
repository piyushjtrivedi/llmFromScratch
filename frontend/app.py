import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gradio as gr

from src.utils.model_inference import InferenceEngine
from src.models.registry import (get_finetuned_weights_path, load_metrics,
                                  list_available_checkpoints)
from src.utils.config import Model_Configs
from src.utils.plotting import plot_loss_curves, plot_comparison, plot_comparison_table

# Discover all trained checkpoints (full fine-tune and LoRA variants).
# Falls back to a static list if no weights directory exists yet.
_checkpoints = list_available_checkpoints()
AVAILABLE_MODELS = [c["display"] for c in _checkpoints] or [
    "gpt2-small (124M)",
    "gpt2-medium (355M)",
    "gemma3-1b",
]

# Map display name → {model_name, lora} for downstream use
_CHECKPOINT_MAP = {c["display"]: c for c in _checkpoints}

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _detect_device()

# Loaded engines are cached so switching back to a previously used model is instant.
_engines: dict[str, InferenceEngine] = {}

def _get_engine(display_name: str) -> InferenceEngine:
    if display_name not in _engines:
        ck = _CHECKPOINT_MAP.get(display_name)
        if ck:
            model_name = ck["model_name"]
            weights_path = get_finetuned_weights_path(model_name, lora=ck["lora"])
            weights_path = weights_path if os.path.exists(weights_path) else None
        else:
            # Fallback: display_name is the raw model name (no trained weights found)
            model_name = display_name.replace(" [LoRA]", "")
            weights_path = None
        _engines[display_name] = InferenceEngine(model_name, weights_path, DEVICE)
    return _engines[display_name]


# ── Tab 1: Generate ──────────────────────────────────────────────────────────

def respond(model_name: str, instruction: str, max_tokens: int, temperature: float) -> str:
    if not instruction.strip():
        return "Please enter an instruction."
    try:
        engine = _get_engine(model_name)
        return engine.generate(instruction, int(max_tokens), temperature)
    except Exception as e:
        return f"Error: {e}"


# ── Tab 2: Loss Curves ───────────────────────────────────────────────────────

def _load_all_metrics() -> dict:
    result = {}
    for model_name in Model_Configs:
        full = load_metrics(model_name, lora=False)
        lora = load_metrics(model_name, lora=True)
        if full is not None or lora is not None:
            result[model_name] = {"full": full, "lora": lora}
    return result


def show_comparison(selection: str):
    """
    "All"         → styled master comparison table across all models.
    <model name>  → side-by-side LoRA vs Full loss curves for that model.
    """
    all_metrics = _load_all_metrics()
    if not all_metrics:
        return None, "No trained checkpoints found. Run training first."

    if selection == "All":
        fig  = plot_comparison_table(all_metrics)
        n    = len(all_metrics)
        full_n = sum(1 for v in all_metrics.values() if v["full"])
        lora_n = sum(1 for v in all_metrics.values() if v["lora"])
        info = f"{n} model(s) · {full_n} full run(s) · {lora_n} LoRA run(s)"
    else:
        if selection not in all_metrics:
            return None, f"No metrics found for **{selection}**. Train the model first."
        fig  = plot_comparison({selection: all_metrics[selection]})
        v    = all_metrics[selection]
        tags = []
        if v["full"]: tags.append("Full")
        if v["lora"]: tags.append("LoRA")
        info = f"**{selection}** — {' + '.join(tags)}"

    return fig, info


def show_loss_curves(display_name: str):
    ck = _CHECKPOINT_MAP.get(display_name)
    if ck:
        model_name, is_lora = ck["model_name"], ck["lora"]
    else:
        model_name, is_lora = display_name.replace(" [LoRA]", ""), False

    metrics = load_metrics(model_name, lora=is_lora)
    if metrics is None:
        return (
            None,
            f"No training metrics found for **{display_name}**. "
            "Run `python -m src.main --model \"<model>\"` to train and save metrics."
        )

    fig = plot_loss_curves(metrics)
    lora_tag = " · LoRA" if is_lora else ""
    info = (
        f"**{display_name}** — "
        f"{metrics.get('num_epochs', '?')} epochs · "
        f"batch size {metrics.get('batch_size', '?')} · "
        f"lr {metrics.get('learning_rate', '?')} · "
        f"trained in {metrics.get('execution_time_minutes', '?')} min{lora_tag}"
    )
    return fig, info


# ── Layout ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="LLM From Scratch") as demo:
    gr.Markdown(f"## LLM From Scratch — Multi-Model Demo\nRunning on: **{DEVICE.upper()}**")

    with gr.Tabs():

        with gr.Tab("Generate"):
            with gr.Row():
                gen_model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0],
                    label="Model",
                    scale=1,
                )
            with gr.Row():
                instruction_box = gr.Textbox(
                    label="Instruction",
                    lines=4,
                    placeholder="Enter your instruction here...",
                    scale=3,
                )
            with gr.Row():
                max_tokens_slider = gr.Slider(50, 512, value=200, step=10, label="Max tokens")
                temperature_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            submit_btn = gr.Button("Generate", variant="primary")
            output_box = gr.Textbox(label="Response", lines=8)

            submit_btn.click(
                fn=respond,
                inputs=[gen_model_selector, instruction_box, max_tokens_slider, temperature_slider],
                outputs=output_box,
            )

        with gr.Tab("Compare Models"):
            with gr.Row():
                compare_selector = gr.Dropdown(
                    choices=["All"] + list(Model_Configs.keys()),
                    value="All",
                    label="View",
                    scale=1,
                )
            compare_info = gr.Markdown()
            compare_plot = gr.Plot(label="Comparison")
            compare_selector.change(
                fn=show_comparison,
                inputs=[compare_selector],
                outputs=[compare_plot, compare_info],
            )

        with gr.Tab("Loss Curves"):
            with gr.Row():
                curve_model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0],
                    label="Model",
                    scale=1,
                )
                load_btn = gr.Button("Load", variant="primary", scale=0)
            curve_info = gr.Markdown()
            loss_plot = gr.Plot(label="Training vs Validation Loss")

            # Load on button click
            load_btn.click(
                fn=show_loss_curves,
                inputs=[curve_model_selector],
                outputs=[loss_plot, curve_info],
            )
            # Also reload automatically when model selection changes
            curve_model_selector.change(
                fn=show_loss_curves,
                inputs=[curve_model_selector],
                outputs=[loss_plot, curve_info],
            )

    demo.load(
        fn=lambda: show_comparison("All"),
        inputs=[],
        outputs=[compare_plot, compare_info],
    )

demo.launch()
