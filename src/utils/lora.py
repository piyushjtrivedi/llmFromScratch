import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Q and V projections — minimal LoRA config from the original paper, shared by GPT-2 and Gemma3.
_DEFAULT_TARGET_MODULES = ("W_query", "W_value")


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a trainable low-rank delta.

    Forward pass: y = W x + (alpha/r) * B A x
    where W is the frozen original weight, A ∈ R^{r × d_in} and B ∈ R^{d_out × r}
    are the trainable LoRA matrices.

    A is initialized with Kaiming uniform; B is initialized to zero so the
    adapter output is exactly zero at the start of training — the model begins
    in the same state as the pretrained checkpoint.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r
        self._lora_enabled = True

        in_features = linear.in_features
        out_features = linear.out_features
        dtype = linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.linear(x)
        if self._lora_enabled:
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            out = out + lora_out * self.scaling
        return out


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: tuple = _DEFAULT_TARGET_MODULES,
) -> nn.Module:
    """
    Apply LoRA to a model in-place.

    Freezes all existing parameters, then wraps every nn.Linear whose attribute
    name appears in target_modules with a LoRALinear. After this call only the
    LoRA A/B matrices have requires_grad=True.

    Args:
        model:          The model to adapt (modified in-place).
        r:              LoRA rank. Higher rank → more capacity, more parameters.
                        Typical values: 4, 8, 16.
        alpha:          LoRA scaling factor. Effective scaling = alpha / r.
                        Setting alpha == r keeps the scaling at 1.0.
        target_modules: Attribute names of nn.Linear layers to wrap.
                        Must match the actual attribute names in the model.
                        Default targets Q and V projections in both GPT-2 and Gemma3.
    Returns:
        The same model instance (for chaining).
    """
    # Freeze everything first so only the LoRA matrices we add remain trainable.
    for param in model.parameters():
        param.requires_grad_(False)

    wrapped = 0
    for module in model.modules():
        for attr in target_modules:
            if hasattr(module, attr):
                child = getattr(module, attr)
                if isinstance(child, nn.Linear):
                    setattr(module, attr, LoRALinear(child, r=r, alpha=alpha))
                    wrapped += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[LoRA] Wrapped {wrapped} layer(s) | r={r} alpha={alpha} | "
        f"trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def enable_lora(model: nn.Module) -> None:
    """Re-enable the LoRA delta in all LoRALinear layers (active by default)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module._lora_enabled = True


def disable_lora(model: nn.Module) -> None:
    """
    Disable the LoRA delta so the model behaves like the frozen base weights.
    Useful for comparing base vs adapted outputs without removing the adapters.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module._lora_enabled = False
