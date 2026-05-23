# Define model configurations in a dictionary for compactness
Model_Configs = {
    "gpt2-small (124M)": {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": True      # Query-key-value bias
    },
    "gpt2-medium (355M)": {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1024,        # Embedding dimension
        "n_heads": 16,         # Number of attention heads
        "n_layers": 24,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": True      # Query-key-value bias
    },
    "gpt2-large (774M)": {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1280,        # Embedding dimension
        "n_heads": 20,         # Number of attention heads
        "n_layers": 36,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": True      # Query-key-value bias
    },
    "gpt2-xl (1558M)": {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1600,        # Embedding dimension
        "n_heads": 25,         # Number of attention heads
        "n_layers": 48,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": True      # Query-key-value bias
    },
    "gemma-2b": {
        "vocab_size": 256000,
        "context_length": 8192,
        "emb_dim": 2048,
        "n_heads": 8,
        "n_kv_heads": 1,       # grouped-query attention
        "n_layers": 18,
        "hidden_dim": 16384,   # SwiGLU intermediate size
        "drop_rate": 0.0,
        "head_dim": 256,
    }
}