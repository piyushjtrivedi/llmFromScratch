import os
import json
import torch
import logging

from src.models.gpt2.gpt import GPTModel
from src.utils.weights_loader.gpt2_weights_loader import GPT2WeightsLoader

from src.models.gemma3.gemma import GemmaModel
from src.utils.weights_loader.gemma3_weights_loader import Gemma3WeightsLoader

from src.utils.colab import get_data_dir

logger = logging.getLogger(__name__)

_MODELS = {
    "gpt2-small (124M)":  GPTModel,
    "gpt2-medium (355M)": GPTModel,
    "gpt2-large (774M)":  GPTModel,
    "gpt2-xl (1558M)":    GPTModel,
    "gemma3-1b":          GemmaModel,
}

_LOADERS = {
    "gpt2-small (124M)":  lambda: GPT2WeightsLoader("124M"),
    "gpt2-medium (355M)": lambda: GPT2WeightsLoader("355M"),
    "gpt2-large (774M)":  lambda: GPT2WeightsLoader("774M"),
    "gpt2-xl (1558M)":    lambda: GPT2WeightsLoader("1558M"),
    "gemma3-1b":          lambda: Gemma3WeightsLoader(),
}

# Subdirectory name within the project data dir — resolved at call time so
# Colab / Drive paths are picked up correctly rather than evaluated once on import.
_FINETUNED_SUBDIR = "fine_tuned_weights"


def _finetuned_dir() -> str:
    return os.path.join(get_data_dir(), _FINETUNED_SUBDIR)


def _sanitize(model_name: str) -> str:
    return model_name.replace(" ", "_").replace("(", "").replace(")", "")


def get_finetuned_weights_path(model_name: str, save_dir: str = None,
                               lora: bool = False) -> str:
    save_dir = save_dir or _finetuned_dir()
    suffix = "_lora" if lora else ""
    return os.path.join(save_dir, f"{_sanitize(model_name)}{suffix}.pth")


def get_model(model_name: str, cfg: dict):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODELS)}")
    return _MODELS[model_name](cfg)


def get_weights_loader(model_name: str):
    if model_name not in _LOADERS:
        raise ValueError(f"No weights loader for '{model_name}'.")
    return _LOADERS[model_name]()


def save_weights(model, model_name: str, save_dir: str = None,
                 lora: bool = False) -> str:
    save_path = get_finetuned_weights_path(model_name, save_dir, lora=lora)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if lora:
        # Save only the trainable adapter params — much smaller than the full model.
        state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    logger.info(f"[Registry] Fine-tuned weights saved to {save_path}")
    return save_path


# ── Metrics (loss curves) ────────────────────────────────────────────────────

def get_metrics_path(model_name: str, save_dir: str = None,
                     lora: bool = False) -> str:
    save_dir = save_dir or _finetuned_dir()
    suffix = "_lora" if lora else ""
    return os.path.join(save_dir, f"{_sanitize(model_name)}{suffix}_metrics.json")


def save_metrics(metrics: dict, model_name: str, save_dir: str = None,
                 lora: bool = False) -> str:
    save_path = get_metrics_path(model_name, save_dir, lora=lora)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[Registry] Training metrics saved to {save_path}")
    return save_path


def load_metrics(model_name: str, save_dir: str = None,
                 lora: bool = False) -> dict | None:
    metrics_path = get_metrics_path(model_name, save_dir, lora=lora)
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if lora and m.get("lora_rank") is None:
        cfg = load_lora_config(model_name, save_dir)
        if cfg:
            m["lora_rank"]  = cfg["r"]
            m["lora_alpha"] = cfg["alpha"]
    return m


# ── LoRA config sidecar ──────────────────────────────────────────────────────

def _lora_config_path(model_name: str, save_dir: str = None) -> str:
    save_dir = save_dir or _finetuned_dir()
    return os.path.join(save_dir, f"{_sanitize(model_name)}_lora_config.json")


def save_lora_config(model_name: str, r: int, alpha: float,
                     target_modules: list, save_dir: str = None) -> str:
    save_path = _lora_config_path(model_name, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"r": r, "alpha": alpha, "target_modules": target_modules}, f)
    logger.info(f"[Registry] LoRA config saved to {save_path}")
    return save_path


def load_lora_config(model_name: str, save_dir: str = None) -> dict | None:
    path = _lora_config_path(model_name, save_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_available_checkpoints(save_dir: str = None) -> list[dict]:
    """
    Scan the fine-tuned weights directory and return all available checkpoints.
    Each entry: {"model_name": str, "lora": bool, "display": str}
    """
    save_dir = save_dir or _finetuned_dir()
    if not os.path.isdir(save_dir):
        return []

    checkpoints = []
    for model_name in _MODELS:
        sanitized = _sanitize(model_name)
        if os.path.exists(os.path.join(save_dir, f"{sanitized}.pth")):
            checkpoints.append({
                "model_name": model_name, "lora": False,
                "display": model_name,
            })
        if os.path.exists(os.path.join(save_dir, f"{sanitized}_lora.pth")):
            checkpoints.append({
                "model_name": model_name, "lora": True,
                "display": f"{model_name} [LoRA]",
            })
    return checkpoints