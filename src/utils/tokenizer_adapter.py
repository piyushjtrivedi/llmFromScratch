class HFTokenizerAdapter:
    """
    Wraps a HuggingFace tokenizer to expose the same interface that tiktoken
    provides, so the rest of the codebase (trainer, collator, inference engine)
    works without any changes regardless of which model is being used.

    Tiktoken interface used in this codebase:
        tokenizer.encode(text, allowed_special={...})  -> list[int]
        tokenizer.decode(token_ids)                    -> str
        tokenizer.eot_token                            -> int  (EOS token id)

    To add support for a new HuggingFace tokenizer, instantiate this class with
    the AutoTokenizer for that model — no other changes needed:
        HFTokenizerAdapter(AutoTokenizer.from_pretrained("some/model"))
    """

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        # Expose eot_token to match tiktoken's attribute name so that the EOS
        # detection in main.py (`hasattr(tokenizer, "eot_token")`) works correctly.
        self.eot_token = hf_tokenizer.eos_token_id

    def encode(self, text, allowed_special=None):
        # allowed_special is tiktoken-specific and controls whether special tokens
        # like <|endoftext|> are tokenised. HF tokenizers handle special tokens
        # differently — we skip adding them here since instruction templates are
        # formatted explicitly in InstructionDataset.format_input().
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        return self._tok.decode(token_ids)
