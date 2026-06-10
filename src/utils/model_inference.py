import torch
import logging

from src.models.registry import get_model, get_weights_loader, load_lora_config
from src.utils.lora import apply_lora, _DEFAULT_TARGET_MODULES
from src.utils.config import Model_Configs
from src.training.trainer import ModelTrainer
from src.data.instruction_dataset import InstructionDataset

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model_name: str, weights_path: str = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)

        self.model = get_model(model_name, Model_Configs[model_name])

        if weights_path:
            if "_lora" in weights_path:
                # Always start from pretrained base weights — the checkpoint only
                # contains adapter deltas (new format) or full state (old format).
                get_weights_loader(model_name)(self.model)

                sd = torch.load(weights_path, map_location=self.device, weights_only=True)

                lora_cfg = load_lora_config(model_name)
                if lora_cfg is None:
                    lora_a_keys = [k for k in sd if k.endswith(".lora_A")]
                    r = sd[lora_a_keys[0]].shape[0] if lora_a_keys else 8
                    lora_cfg = {"r": r, "alpha": 16.0,
                                "target_modules": list(_DEFAULT_TARGET_MODULES)}
                    logger.warning(
                        f"[InferenceEngine] No lora_config.json — inferred r={r}, alpha=16.0"
                    )

                apply_lora(self.model, r=lora_cfg["r"], alpha=lora_cfg["alpha"],
                           target_modules=tuple(lora_cfg["target_modules"]))
                logger.info(f"[InferenceEngine] Applied LoRA r={lora_cfg['r']} alpha={lora_cfg['alpha']}")

                # New format: adapter-only keys (strict=False keeps pretrained base).
                # Old format: full state dict — strict=True overwrites base weights too.
                adapter_only = all("lora_" in k for k in sd)
                self.model.load_state_dict(sd, strict=not adapter_only)
            else:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device,
                                                       weights_only=True))
            logger.info(f"[InferenceEngine] Loaded fine-tuned weights from {weights_path}")
        else:
            get_weights_loader(model_name)(self.model)
            logger.info(f"[InferenceEngine] Loaded pretrained weights for {model_name}")

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.get_tokenizer()
        self.context_size = self.model.get_context_size()
        self.eos_id = self._get_eos_id()

    def _get_eos_id(self):
        # tiktoken (GPT-2) exposes eot_token; HuggingFace tokenizers expose eos_token_id
        if hasattr(self.tokenizer, "eot_token"):
            return self.tokenizer.eot_token          # GPT-2: 50256
        if hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id       # Gemma: 1
        return None

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7, top_k: int = 50) -> str:
        input_text = InstructionDataset.format_input({"instruction": prompt, "input": ""})

        allowed = {"<|endoftext|>"} if hasattr(self.tokenizer, "eot_token") else set()
        encoded = self.tokenizer.encode(input_text, allowed_special=allowed)
        idx = torch.tensor(encoded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            token_ids = ModelTrainer.generate_text_simple(
                model=self.model,
                idx=idx,
                max_new_tokens=max_new_tokens,
                context_size=self.context_size,
                temperature=temperature,
                top_k=top_k,
                eos_id=self.eos_id,
            )

        output = self.tokenizer.decode(token_ids.squeeze(0).tolist())
        return output[len(input_text):].replace("### Response:", "").strip()
