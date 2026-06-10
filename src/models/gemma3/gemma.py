import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os

from src.models.base_model import BaseLanguageModel
from src.utils.tokenizer_adapter import HFTokenizerAdapter
from src.models.gemma3.normalization import RMSNorm
from src.models.gemma3.transformer import TransformerBlock
from src.models.gemma3.rope_wrapper import RoPE

from src.utils.colab import get_data_dir


class GemmaModel(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)for attn_type in cfg["layer_types"]
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.cfg = cfg

        # Reusuable utilities
        cos_local, sin_local = RoPE.compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = RoPE.compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

        # Cache masks by (seq_len, device) — they are pure functions of these two
        # values so there is no reason to rebuild them on every forward pass.
        self._mask_cache: dict = {}

    def _create_masks(self, seq_len, device):
        key = (seq_len, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past

        self._mask_cache[key] = (mask_global, mask_local)
        return mask_global, mask_local

    def forward(self, input_ids):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        return logits
    
    def get_context_size(self) -> int:
        # Gemma uses RoPE (no learned positional embedding), so context size
        # comes from the config rather than a pos_emb weight matrix.
        return self.cfg["context_length"]

    def get_tokenizer(self):
        tokenizer_id = self.cfg.get("tokenizer_id", "gpt2")

        if tokenizer_id == "gpt2":
            # Custom models trained with GPT-2 vocab (vocab_size=50257) use tiktoken.
            # No HuggingFace auth required.
            return tiktoken.get_encoding("gpt2")

        # Any other tokenizer_id is treated as a HuggingFace model repo.
        # The HFTokenizerAdapter wraps it to expose the same interface as tiktoken
        # (encode / decode / eot_token) so no other code needs to change.
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace tokenizers. "
                "Install with: pip install transformers"
            )

        cache_dir = os.path.join(get_data_dir(), "gemma3_cache")

        # Try local cache first — no network call if tokenizer was already downloaded.
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, cache_dir=cache_dir, local_files_only=True
            )
            return HFTokenizerAdapter(hf_tokenizer)
        except Exception:
            pass  # cache miss — fall through to download

        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=cache_dir)
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower() or "credentials" in str(e).lower():
                raise PermissionError(
                    f"\n\n401 Unauthorized — '{tokenizer_id}' is a gated model.\n"
                    "Complete these steps once, then re-run:\n"
                    f"  1. Accept the license at https://huggingface.co/{tokenizer_id}\n"
                    "  2. Create a token at https://huggingface.co/settings/tokens\n"
                    "  3. huggingface-cli login   (paste your token when prompted)\n"
                ) from e
            raise
        return HFTokenizerAdapter(hf_tokenizer)