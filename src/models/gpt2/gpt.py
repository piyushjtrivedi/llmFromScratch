import torch
from torch import nn
import tiktoken

from src.models.base_model import BaseLanguageModel
from src.models.gpt2.transformer import TransformerBlock
from src.models.gpt2.normalization import LayerNorm


class GPTModel(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__()
        # Initialize token instances
        # Token Embeddgings has size of vocabulary size * embeddings dimensions
        # Position Embeddings has size of context length * embeddings dimensions
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Initialize transformer blocks sequentially
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Initialize final normalization block
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Initialize the out head provides logit of token against vocabulary size
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Convert text to tokens embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Generate positional embeddings for the tokens
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Input embeddings = token embeddings + positional embeddings
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]

        # Apply dropout function to the input batch and tokens
        x = self.drop_emb(x)
        
        # Feed the sequence to Transformer blocks. No of sequential bloacks decided by n_layers
        x = self.trf_blocks(x)
        
        # Feed theout from final transformer bloack for a final normalization
        x = self.final_norm(x)
        
        # Feed these to out head which give give logits of token againsts vocabulary size
        logits = self.out_head(x)
        
        # Return these logits
        return logits

    def get_context_size(self) -> int:
        return self.pos_emb.weight.shape[0]

    def get_tokenizer(self):
        return tiktoken.get_encoding("gpt2")