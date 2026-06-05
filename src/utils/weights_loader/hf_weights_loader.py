import os
import logging
from abc import abstractmethod

import torch

from src.utils.weights_loader.base_weights_loader import BaseWeightsLoader
from src.utils.colab import get_data_dir

logger = logging.getLogger(__name__)


class HFWeightsLoader(BaseWeightsLoader):
    """
    Abstract base class for loading pretrained weights from HuggingFace Hub
    into a custom model in this codebase.

    Responsibilities of this class:
    - Download the checkpoint from HuggingFace Hub (with local caching).
    - Load all weight shards into a single state dict.
    - Call _map_keys() to translate HF key names to our model's naming.
    - Load the mapped dict into the model via load_state_dict().

    To add support for a new model from HuggingFace, subclass this and
    implement _map_keys() with the key translation for that architecture:

        class MyModelWeightsLoader(HFWeightsLoader):
            def __init__(self):
                super().__init__("org/model-name", cache_dir="data/mymodel_cache")

            def _map_keys(self, hf_state_dict, model):
                mapped = {}
                mapped["tok_emb.weight"] = hf_state_dict["model.embed_tokens.weight"]
                # ... rest of the mapping ...
                return mapped

    Prerequisites (same for all subclasses):
        pip install huggingface_hub safetensors
        huggingface-cli login   # required for gated models (e.g. Gemma)
    """

    def __init__(self, repo_id: str, cache_dir: str = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.join(get_data_dir(), "hf_cache")

    def _download(self) -> dict:
        """
        Download the model checkpoint from HuggingFace Hub and return a single
        merged state dict. Uses local disk cache so repeated calls are instant.
        Supports both .safetensors (preferred) and legacy .bin shards.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        _ignore = ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"]

        # Try the local cache first — no network call, instant if already downloaded.
        try:
            local_dir = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                ignore_patterns=_ignore,
            )
            logger.info(f"[HFWeightsLoader] Loaded '{self.repo_id}' from local cache")
            return self._load_shards(local_dir)
        except Exception:
            # Cache miss — weights not downloaded yet, fall through to fetch.
            pass

        # First-time download from HuggingFace Hub.
        logger.info(f"[HFWeightsLoader] Downloading '{self.repo_id}' → '{self.cache_dir}' ...")
        try:
            local_dir = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=self.cache_dir,
                ignore_patterns=_ignore,
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower() or "credentials" in str(e).lower():
                raise PermissionError(
                    f"\n\n401 Unauthorized — '{self.repo_id}' is a gated model.\n"
                    "Complete these steps once, then re-run:\n"
                    f"  1. Accept the license at https://huggingface.co/{self.repo_id}\n"
                    "  2. Create a token at https://huggingface.co/settings/tokens\n"
                    "  3. huggingface-cli login   (paste your token when prompted)\n"
                ) from e
            raise
        logger.info(f"[HFWeightsLoader] Checkpoint available at: {local_dir}")
        return self._load_shards(local_dir)

    def _load_shards(self, local_dir: str) -> dict:
        """Merge all weight shards in local_dir into one dict."""
        state_dict = {}

        safetensors_files = sorted(
            f for f in os.listdir(local_dir) if f.endswith(".safetensors")
        )
        bin_files = sorted(
            f for f in os.listdir(local_dir)
            if f.endswith(".bin") and "optimizer" not in f
        )

        if safetensors_files:
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "safetensors is required. Install with: pip install safetensors"
                )
            for fname in safetensors_files:
                state_dict.update(load_file(os.path.join(local_dir, fname), device="cpu"))
            logger.info(f"[HFWeightsLoader] Loaded {len(safetensors_files)} .safetensors shard(s)")

        elif bin_files:
            for fname in bin_files:
                state_dict.update(
                    torch.load(os.path.join(local_dir, fname), map_location="cpu", weights_only=True)
                )
            logger.info(f"[HFWeightsLoader] Loaded {len(bin_files)} .bin shard(s)")

        else:
            raise FileNotFoundError(
                f"No weight files (.safetensors or .bin) found in {local_dir}. "
                "Check that the download completed successfully."
            )

        return state_dict

    @abstractmethod
    def _map_keys(self, hf_state_dict: dict, model) -> dict:
        """
        Translate HuggingFace checkpoint key names to this codebase's naming.

        Args:
            hf_state_dict: raw state dict loaded from the HuggingFace checkpoint
            model: the target model instance (available for shape inspection)
        Returns:
            dict with keys matching model.state_dict()
        """
        ...

    def __call__(self, model):
        hf_state_dict = self._download()
        mapped = self._map_keys(hf_state_dict, model)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning(f"[HFWeightsLoader] Keys not found in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"[HFWeightsLoader] Checkpoint keys not used (ignored): {unexpected}")
        logger.info(f"[HFWeightsLoader] '{self.repo_id}' weights loaded successfully")
