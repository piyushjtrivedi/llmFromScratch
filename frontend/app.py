import sys
import os
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.model_inference import InferenceEngine
from src.models.registry import get_finetuned_weights_path

# Models available in the UI. Add new entries here as weights are trained.
AVAILABLE_MODELS = [
    "gpt2-small (124M)",
    # "gpt2-medium (355M)",
    # "gemma-2b",
]

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _detect_device()

# Loaded engines are cached so switching back to a previously used model is instant.
_engines: dict[str, InferenceEngine] = {}

def _get_engine(model_name: str) -> InferenceEngine:
    if model_name not in _engines:
        weights_path = get_finetuned_weights_path(model_name)
        weights_path = weights_path if os.path.exists(weights_path) else None
        _engines[model_name] = InferenceEngine(model_name, weights_path, DEVICE)
    return _engines[model_name]


def respond(model_name: str, instruction: str, max_tokens: int, temperature: float) -> str:
    if not instruction.strip():
        return "Please enter an instruction."
    try:
        engine = _get_engine(model_name)
        return engine.generate(instruction, int(max_tokens), temperature)
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="LLM From Scratch") as demo:
    gr.Markdown(f"## LLM From Scratch — Multi-Model Demo\nRunning on: **{DEVICE.upper()}**")

    with gr.Row():
        model_selector = gr.Dropdown(
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
        inputs=[model_selector, instruction_box, max_tokens_slider, temperature_slider],
        outputs=output_box,
    )

demo.launch()
