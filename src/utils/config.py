import torch
# Define model configurations in a dictionary for compactness.
# batch_size and allowed_max_length are conservative defaults tuned for ~18 GB MPS.
# Pass --batch-size or --max-length on the CLI to override.
Model_Configs = {
    "gpt2-small (124M)": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
        # Training memory defaults (~2 GB model + optimizer states)
        # effective batch = batch_size × gradient_accumulation_steps = 8
        "batch_size": 8,
        "allowed_max_length": 128,
        "gradient_accumulation_steps": 1,
    },
    "gpt2-medium (355M)": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True,
        # Training memory defaults (~5.7 GB model + optimizer states)
        # micro batch=1, seq=64: attention scores are O(seq²) so halving seq saves 4× that tensor
        # effective batch = 1×4 = 4
        "batch_size": 1,
        "allowed_max_length": 64,
        "gradient_accumulation_steps": 4,
    },
    "gpt2-large (774M)": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": True,
        # Training memory defaults (~12.4 GB model + optimizer states)
        # effective batch = 1×4 = 4
        "batch_size": 1,
        "allowed_max_length": 64,
        "gradient_accumulation_steps": 4,
    },
    "gpt2-xl (1558M)": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.1,
        "qkv_bias": True,
        # Training memory defaults (~24 GB — inference only on 18 GB MPS)
        "batch_size": 1,
        "allowed_max_length": 32,
        "gradient_accumulation_steps": 1,
    },

    # Official Google Gemma3-1B loaded from HuggingFace Hub.
    # Architecture dimensions match google/gemma-3-1b exactly so pretrained
    # weights transfer without any tensor reshaping.
    # Requires: pip install huggingface_hub safetensors transformers
    #           huggingface-cli login   (Gemma is a gated model)
    "gemma3-1b": {
        "vocab_size": 262144,
        "context_length": 32_768,
        "emb_dim": 1152,
        "n_heads": 4,
        "n_kv_groups": 1,
        "n_layers": 26,
        "hidden_dim": 6912,
        "head_dim": 256,
        "qk_norm": True,
        "query_pre_attn_scalar": 256,
        "rope_local_base": 10_000.0,
        "rope_base": 1_000_000.0,
        "sliding_window": 512,
        "layer_types": [
            # Pattern: 5 sliding → 1 full, repeated 4 times, then 2 trailing sliding
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention",
        ],
        "dtype": torch.bfloat16,
        "drop_rate": 0.0,
        "tokenizer_id": "google/gemma-3-1b-pt",  # HuggingFace tokenizer (vocab=262144)
        # Conservative MPS memory defaults for a ~2.3 GB bf16 model
        "batch_size": 1,
        "allowed_max_length": 64,
        "gradient_accumulation_steps": 8,
    },
}