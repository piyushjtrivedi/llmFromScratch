from abc import ABC, abstractmethod
import torch.nn as nn

class BaseLanguageModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def get_context_size(self) -> int:
        """Return the max sequence length this model supports."""
        ...

    @abstractmethod
    def get_tokenizer(self):
        """Return the tokenizer instance appropriate for this model."""
        ...