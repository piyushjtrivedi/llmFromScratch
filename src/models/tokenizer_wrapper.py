import torch
import tiktoken

class TokenizerWrapper:
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def text_to_token_ids(self,text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0) # remove batch dimension
        return self.tokenizer.decode(flat.tolist())