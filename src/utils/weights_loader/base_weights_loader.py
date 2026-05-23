from abc import ABC, abstractmethod

class BaseWeightsLoader(ABC):
    @abstractmethod
    def __call__(self, model):
        """Load pretrained weights into model in-place."""
        ...