"""Convert raw attention maps to visualizable heatmaps."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional

_SOT = "<|" + "startoftext" + "|>"
_EOT = "<|" + "endoftext" + "|>"
DEFAULT_SKIP_TOKENS = {_SOT, _EOT, ""}


def normalize_map(attn_map: torch.Tensor) -> np.ndarray:
    """Normalize a 2D attention map to [0, 1]."""
    m = attn_map.float()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.numpy()


def upscale_map(attn_map: np.ndarray, target_size: tuple) -> np.ndarray:
    """Bicubic upscale attention map to target (height, width)."""
    tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
    upscaled = F.interpolate(tensor, size=target_size, mode='bicubic', align_corners=False)
    return upscaled.squeeze().numpy()


def overlay_heatmap(image: Image.Image, attn_map: np.ndarray, alpha: float = 0.55, colormap: str = 'hot') -> Image.Image:
    """Overlay a normalized attention heatmap on an image."""
    w, h = image.size
    attn_up = np.clip(upscale_map(attn_map, (h, w)), 0, 1)

    cmap = cm.get_cmap(colormap)
    heatmap_rgba = (cmap(attn_up) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgba).convert("RGBA")

    image_rgba = image.convert("RGBA")
    blended = Image.blend(image_rgba, heatmap_img, alpha=alpha)
    return blended.convert("RGB")


def visualize_token_maps(image, token_maps, tokens, skip_tokens=None, cols=4, save_path=None):
    """Grid visualization: original image + one heatmap per meaningful token."""
    if skip_tokens is None:
        skip_tokens = DEFAULT_SKIP_TOKENS

    display = [
        (i, tok) for i, tok in enumerate(tokens)
        if tok not in skip_tokens and not tok.startswith("<")
    ]

    n = len(display) + 1
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    axes[0].imshow(image)
    axes[0].set_title("Generated image", fontsize=11, pad=6)
    axes[0].axis('off')

    for plot_idx, (token_idx, token) in enumerate(display, start=1):
        if plot_idx >= len(axes):
            break
        attn = normalize_map(token_maps[token_idx])
        overlaid = overlay_heatmap(image, attn)
        axes[plot_idx].imshow(overlaid)
        title_str = '"' + token + '"'
        axes[plot_idx].set_title(title_str, fontsize=11, pad=6)
        axes[plot_idx].axis('off')

    for i in range(len(display) + 1, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Cross-attention grounding maps", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
