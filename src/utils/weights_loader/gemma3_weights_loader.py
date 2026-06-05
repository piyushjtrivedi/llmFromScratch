import logging
import os

from src.utils.weights_loader.hf_weights_loader import HFWeightsLoader
from src.utils.colab import get_data_dir

logger = logging.getLogger(__name__)

_GEMMA3_1B_REPO = "google/gemma-3-1b-pt"


class Gemma3WeightsLoader(HFWeightsLoader):
    """
    Downloads google/gemma-3-1b-pt from HuggingFace Hub and maps its weights
    into our custom GemmaModel.

    Why the key mapping is needed:
    HuggingFace Gemma3 uses its own parameter naming convention (e.g.
    model.layers.{i}.self_attn.q_proj.weight) while our model uses a simpler
    convention (blocks.{i}.att.W_query.weight). This class translates between
    the two so load_state_dict() works correctly.

    Key mapping (HuggingFace → our model):
    ─────────────────────────────────────────────────────────────────────────
    model.embed_tokens.weight                         → tok_emb.weight
    model.norm.weight                                 → final_norm.scale
    lm_head.weight                                    → out_head.weight

    Per layer (i = 0 … n_layers-1):
    model.layers.{i}.self_attn.q_proj.weight         → blocks.{i}.att.W_query.weight
    model.layers.{i}.self_attn.k_proj.weight         → blocks.{i}.att.W_key.weight
    model.layers.{i}.self_attn.v_proj.weight         → blocks.{i}.att.W_value.weight
    model.layers.{i}.self_attn.o_proj.weight         → blocks.{i}.att.out_proj.weight
    model.layers.{i}.self_attn.q_norm.weight         → blocks.{i}.att.q_norm.scale
    model.layers.{i}.self_attn.k_norm.weight         → blocks.{i}.att.k_norm.scale
    model.layers.{i}.mlp.gate_proj.weight            → blocks.{i}.ff.fc1.weight
    model.layers.{i}.mlp.up_proj.weight              → blocks.{i}.ff.fc2.weight
    model.layers.{i}.mlp.down_proj.weight            → blocks.{i}.ff.fc3.weight
    model.layers.{i}.input_layernorm.weight          → blocks.{i}.input_layernorm.scale
    model.layers.{i}.post_attention_layernorm.weight → blocks.{i}.post_attention_layernorm.scale
    model.layers.{i}.pre_feedforward_layernorm.weight  → blocks.{i}.pre_feedforward_layernorm.scale
    model.layers.{i}.post_feedforward_layernorm.weight → blocks.{i}.post_feedforward_layernorm.scale

    RMSNorm note:
    Both HuggingFace and our RMSNorm initialise weights to zeros and apply
    (1 + weight) during the forward pass, so weights transfer directly with no
    offset adjustment needed.

    Gemma3 ties lm_head to the embedding matrix. If lm_head.weight is absent
    from the checkpoint, we reuse tok_emb.weight as the output projection.

    Prerequisites:
        pip install huggingface_hub safetensors
        huggingface-cli login   # Gemma is a gated model — accept terms on HF first
    """

    def __init__(self, repo_id: str = _GEMMA3_1B_REPO):
        super().__init__(repo_id=repo_id, cache_dir=os.path.join(get_data_dir(), "gemma3_cache"))

    def _map_keys(self, hf_state_dict: dict, model) -> dict:
        mapped = {}
        n_layers = len(model.blocks)

        # ── Embeddings and output head ───────────────────────────────────────
        mapped["tok_emb.weight"] = hf_state_dict["model.embed_tokens.weight"]
        mapped["final_norm.scale"] = hf_state_dict["model.norm.weight"]
        # Gemma3 ties lm_head weights to the input embedding (weight tying).
        # The checkpoint may omit lm_head.weight; fall back to the embedding.
        mapped["out_head.weight"] = hf_state_dict.get(
            "lm_head.weight", hf_state_dict["model.embed_tokens.weight"]
        )

        # ── Per-layer weights ────────────────────────────────────────────────
        for i in range(n_layers):
            hf = f"model.layers.{i}"       # HuggingFace prefix
            ours = f"blocks.{i}"           # our model prefix

            # Attention projections
            mapped[f"{ours}.att.W_query.weight"]  = hf_state_dict[f"{hf}.self_attn.q_proj.weight"]
            mapped[f"{ours}.att.W_key.weight"]    = hf_state_dict[f"{hf}.self_attn.k_proj.weight"]
            mapped[f"{ours}.att.W_value.weight"]  = hf_state_dict[f"{hf}.self_attn.v_proj.weight"]
            mapped[f"{ours}.att.out_proj.weight"] = hf_state_dict[f"{hf}.self_attn.o_proj.weight"]

            # QK normalisation (present when qk_norm=True in config)
            if f"{hf}.self_attn.q_norm.weight" in hf_state_dict:
                mapped[f"{ours}.att.q_norm.scale"] = hf_state_dict[f"{hf}.self_attn.q_norm.weight"]
                mapped[f"{ours}.att.k_norm.scale"] = hf_state_dict[f"{hf}.self_attn.k_norm.weight"]

            # Feed-forward (SwiGLU: gate × up → down)
            mapped[f"{ours}.ff.fc1.weight"] = hf_state_dict[f"{hf}.mlp.gate_proj.weight"]
            mapped[f"{ours}.ff.fc2.weight"] = hf_state_dict[f"{hf}.mlp.up_proj.weight"]
            mapped[f"{ours}.ff.fc3.weight"] = hf_state_dict[f"{hf}.mlp.down_proj.weight"]

            # Layer norms — HF stores as "weight", our RMSNorm stores as "scale"
            mapped[f"{ours}.input_layernorm.scale"]            = hf_state_dict[f"{hf}.input_layernorm.weight"]
            mapped[f"{ours}.post_attention_layernorm.scale"]   = hf_state_dict[f"{hf}.post_attention_layernorm.weight"]
            mapped[f"{ours}.pre_feedforward_layernorm.scale"]  = hf_state_dict[f"{hf}.pre_feedforward_layernorm.weight"]
            mapped[f"{ours}.post_feedforward_layernorm.scale"] = hf_state_dict[f"{hf}.post_feedforward_layernorm.weight"]

        return mapped
