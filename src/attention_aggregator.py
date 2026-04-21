"""
Paper-grounded attention map aggregation strategies.

Implements methods from DAAM, Prompt-to-Prompt, and Attend-and-Excite,
verified against their official GitHub repos.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


class GaussianSmoothing(nn.Module):
    """2D Gaussian smoothing via depthwise conv (from Attend-and-Excite)."""

    def __init__(self, channels: int = 1, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(sigma, (int, float)):
            sigma = [float(sigma), float(sigma)]

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing='ij',
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi))) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.pad = kernel_size[0] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(x, weight=self.weight.to(x.dtype), groups=self.groups)


@dataclass
class AggregationConfig:
    """Configuration for attention map aggregation strategy."""
    name: str = "daam_baseline"
    daam_all_res: bool = False
    p2p_late_timestep_fraction: float = 0.0
    ae_gaussian_smooth: bool = False
    ae_gaussian_sigma: float = 0.5
    ae_gaussian_kernel: int = 3
    compound_eval: bool = False


ABLATION_CONFIGS = {
    "P1_daam": AggregationConfig(
        name="P1: DAAM (Tang et al.)",
        daam_all_res=True,
    ),
    "P2_p2p": AggregationConfig(
        name="P2: Prompt-to-Prompt (Hertz et al.)",
        p2p_late_timestep_fraction=0.5,
    ),
    "P3_ae": AggregationConfig(
        name="P3: Attend-and-Excite (Chefer et al.)",
        ae_gaussian_smooth=True,
        ae_gaussian_sigma=0.5,
        ae_gaussian_kernel=3,
    ),
    "ours": AggregationConfig(
        name="Ours (DAAM+P2P+A&E+Compound)",
        daam_all_res=True,
        p2p_late_timestep_fraction=0.5,
        ae_gaussian_smooth=True,
        ae_gaussian_sigma=0.5,
        ae_gaussian_kernel=3,
        compound_eval=True,
    ),
}


class AttentionAggregator:
    """Aggregates raw attention maps using paper-grounded strategies."""

    def __init__(self, config: AggregationConfig):
        self.config = config
        self._smoother = None
        if config.ae_gaussian_smooth:
            self._smoother = GaussianSmoothing(
                channels=1,
                kernel_size=config.ae_gaussian_kernel,
                sigma=config.ae_gaussian_sigma,
            )

    def aggregate(self, raw_storage: dict) -> torch.Tensor:
        if self.config.daam_all_res:
            return self._aggregate_daam(raw_storage)
        return self._aggregate_single_res(raw_storage, resolution=16)

    def _aggregate_daam(self, raw_storage: dict) -> torch.Tensor:
        """DAAM: bicubic upscale all resolutions to 64x64, clamp≥0, mean."""
        latent_side = 64
        all_upscaled = []

        for layer_name, maps in raw_storage.items():
            hw = maps[0].shape[0]
            side = int(hw ** 0.5)

            stacked = torch.stack(maps, dim=0).float()
            stacked = self._maybe_filter_late_steps(stacked)
            mean_map = stacked.mean(dim=0)

            seq_len = mean_map.shape[1]
            mean_map = mean_map.reshape(side, side, seq_len)
            mean_map = mean_map.permute(2, 0, 1).unsqueeze(0)

            upscaled = F.interpolate(
                mean_map, size=(latent_side, latent_side),
                mode='bicubic', align_corners=False,
            ).clamp_(min=0)
            all_upscaled.append(upscaled)

        if not all_upscaled:
            raise ValueError("No attention maps found in raw storage")

        result = torch.cat(all_upscaled, dim=0).mean(dim=0)

        if self.config.ae_gaussian_smooth and self._smoother is not None:
            result = self._apply_ae_smoothing(result)

        return result

    def _aggregate_single_res(self, raw_storage: dict, resolution: int = 16) -> torch.Tensor:
        """P2P/A&E: filter by resolution, sum/count across layers."""
        all_maps = []

        for layer_name, maps in raw_storage.items():
            hw = maps[0].shape[0]
            side = int(hw ** 0.5)
            if side != resolution:
                continue

            stacked = torch.stack(maps, dim=0).float()
            stacked = self._maybe_filter_late_steps(stacked)
            mean_map = stacked.mean(dim=0).reshape(resolution, resolution, -1)
            all_maps.append(mean_map)

        if not all_maps:
            available = self._get_available_resolutions(raw_storage)
            if available:
                closest = min(available, key=lambda r: abs(r - resolution))
                return self._aggregate_single_res(raw_storage, resolution=closest)
            raise ValueError("No attention maps found")

        combined = torch.stack(all_maps, dim=0).sum(dim=0) / len(all_maps)
        result = combined.permute(2, 0, 1)

        if self.config.ae_gaussian_smooth and self._smoother is not None:
            result = self._apply_ae_smoothing(result)

        return result

    def _maybe_filter_late_steps(self, stacked: torch.Tensor) -> torch.Tensor:
        """P2P: keep only the last fraction of timesteps."""
        frac = self.config.p2p_late_timestep_fraction
        if frac <= 0 or frac >= 1.0:
            return stacked
        T = stacked.shape[0]
        return stacked[int(T * (1.0 - frac)):]

    def _apply_ae_smoothing(self, token_maps: torch.Tensor) -> torch.Tensor:
        """A&E: 2D Gaussian conv (σ=0.5, k=3, reflect pad) per token map."""
        smoothed = []
        for i in range(token_maps.shape[0]):
            inp = token_maps[i].unsqueeze(0).unsqueeze(0).float()
            out = self._smoother(inp).squeeze(0).squeeze(0)
            smoothed.append(out)
        return torch.stack(smoothed, dim=0)

    def _get_available_resolutions(self, raw_storage: dict) -> list:
        resolutions = set()
        for layer_name, maps in raw_storage.items():
            resolutions.add(int(maps[0].shape[0] ** 0.5))
        return sorted(resolutions)
