import os
import torch
import logging

from src.models.gpt2.gpt import GPTModel
from src.utils.weights_loader.gpt2_weights_loader import GPT2WeightsLoader

logger = logging.getLogger(__name__)

_MODELS = {
    "gpt2-small (124M)": GPTModel,
    "gpt2-medium (355M)": GPTModel,
    "gpt2-large (774M)": GPTModel,
    "gpt2-xl (1558M)": GPTModel,
    # "gemma-2b": GemmaModel,
}

_LOADERS = {
    "gpt2-small (124M)": lambda: GPT2WeightsLoader("124M"),
    "gpt2-medium (355M)": lambda: GPT2WeightsLoader("355M"),
    "gpt2-large (774M)": lambda: GPT2WeightsLoader("774M"),
    "gpt2-xl (1558M)": lambda: GPT2WeightsLoader("1558M"),
    # "gemma-2b": lambda: GemmaWeightsLoader("2b"),
}

_FINETUNED_WEIGHTS_DIR = "data/fine_tuned_weights"


def _sanitize(model_name: str) -> str:
    return model_name.replace(" ", "_").replace("(", "").replace(")", "")


def get_finetuned_weights_path(model_name: str, save_dir: str = _FINETUNED_WEIGHTS_DIR) -> str:
    return os.path.join(save_dir, f"{_sanitize(model_name)}.pth")


def get_model(model_name: str, cfg: dict):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODELS)}")
    return _MODELS[model_name](cfg)


def get_weights_loader(model_name: str):
    if model_name not in _LOADERS:
        raise ValueError(f"No weights loader for '{model_name}'.")
    return _LOADERS[model_name]()


def save_weights(model, model_name: str, save_dir: str = _FINETUNED_WEIGHTS_DIR) -> str:
    save_path = get_finetuned_weights_path(model_name, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"[Registry] Fine-tuned weights saved to {save_path}")
    return save_path