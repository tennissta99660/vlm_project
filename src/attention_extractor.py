"""Hooks into SD UNet cross-attention layers to collect attention maps during denoising."""
import torch
import numpy as np
from typing import Optional


class AttentionStore:
    """Stores cross-attention maps from all hooked UNet layers."""

    def __init__(self):
        self.storage = {}
        self.step = 0

    def reset(self):
        self.storage = {}
        self.step = 0

    def __call__(self, attn_weights: torch.Tensor, layer_name: str):
        """Called by the hook — averages over heads and stores [H*W, seq_len]."""
        bh, hw, seq = attn_weights.shape
        heads = 8
        attn = attn_weights.reshape(-1, heads, hw, seq)
        attn = attn.mean(dim=1)[0].detach().cpu()

        if layer_name not in self.storage:
            self.storage[layer_name] = []
        self.storage[layer_name].append(attn)
        self.step += 1

    def get_token_maps(self, resolution: int = 16) -> torch.Tensor:
        """Aggregate attention across layers/timesteps at target resolution.
        Returns [seq_len, resolution, resolution]."""
        all_maps = []
        for layer_name, maps in self.storage.items():
            hw = maps[0].shape[0]
            side = int(hw ** 0.5)
            if side != resolution:
                continue
            stacked = torch.stack(maps, dim=0)
            mean_map = stacked.mean(dim=0).reshape(resolution, resolution, -1)
            all_maps.append(mean_map)

        if not all_maps:
            raise ValueError(
                f"No maps at resolution {resolution}. "
                f"Available: {self._available_resolutions()}"
            )

        combined = torch.stack(all_maps, dim=0).mean(dim=0)
        return combined.permute(2, 0, 1)

    def _available_resolutions(self) -> list:
        resolutions = set()
        for maps in self.storage.values():
            resolutions.add(int(maps[0].shape[0] ** 0.5))
        return sorted(resolutions)

    def get_raw_storage(self) -> dict:
        return self.storage

    def save_raw_storage(self, path: str) -> None:
        """Save raw attention storage for ablation re-aggregation."""
        torch.save(self.storage, path)

    @staticmethod
    def load_raw_storage(path: str) -> dict:
        return torch.load(path, map_location="cpu")


def register_attention_hooks(unet, store: AttentionStore) -> list:
    """Register attention-capturing processors on all cross-attention (attn2) layers."""
    hooks = []

    def make_attn_processor(original_processor, layer_name: str):
        class StoringProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, **kwargs):
                result = original_processor(
                    attn, hidden_states, encoder_hidden_states,
                    attention_mask, **kwargs
                )
                if encoder_hidden_states is not None:
                    with torch.no_grad():
                        q = attn.to_q(hidden_states)
                        k = attn.to_k(encoder_hidden_states)
                        q = attn.head_to_batch_dim(q)
                        k = attn.head_to_batch_dim(k)
                        attn_weights = attn.get_attention_scores(q, k, attention_mask)
                        store(attn_weights, layer_name)
                return result
        return StoringProcessor()

    for name, module in unet.named_modules():
        if hasattr(module, 'processor') and 'attn2' in name:
            original = module.processor
            module.processor = make_attn_processor(original, name)
            hooks.append((name, module, original))

    return hooks


def restore_processors(hooks: list) -> None:
    """Restore original attention processors."""
    for name, module, original in hooks:
        module.processor = original
