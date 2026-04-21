"""Helper script to create heatmap.py with special tokens handled properly."""
content = '''"""
Convert raw attention maps to visualizable heatmaps.

Given per-token attention maps from AttentionStore, this module:
1. Normalizes the raw attention values to [0, 1]
2. Upscales from the attention resolution (e.g., 16x16) to image resolution (512x512)
3. Overlays colored heatmaps on the generated image
4. Creates a grid visualization showing all token heatmaps side-by-side
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional

# Build special token names dynamically to avoid literal issues
_SOT = "<|" + "startoftext" + "|>"
_EOT = "<|" + "endoftext" + "|>"
DEFAULT_SKIP_TOKENS = {_SOT, _EOT, ""}


def normalize_map(attn_map: torch.Tensor) -> np.ndarray:
    """
    Normalize a 2D attention map to [0, 1] range.
    
    Args:
        attn_map: 2D tensor of raw attention values
    
    Returns:
        Normalized numpy array with values in [0, 1]
    """
    m = attn_map.float()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.numpy()


def upscale_map(attn_map: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Upscale attention map to image resolution using bicubic interpolation.
    
    Args:
        attn_map: 2D numpy array (e.g., 16x16)
        target_size: (height, width) tuple for output resolution
    
    Returns:
        Upscaled numpy array at target resolution
    """
    tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
    upscaled = F.interpolate(
        tensor, size=target_size, mode='bicubic', align_corners=False
    )
    return upscaled.squeeze().numpy()


def overlay_heatmap(
    image: Image.Image,
    attn_map: np.ndarray,
    alpha: float = 0.55,
    colormap: str = 'hot',
) -> Image.Image:
    """
    Overlay a normalized attention heatmap on an image.
    
    Args:
        image: PIL Image to overlay on
        attn_map: 2D normalized attention map (values in [0, 1])
        alpha: blending factor (higher = more heatmap visible)
        colormap: matplotlib colormap name (e.g., 'hot', 'jet', 'viridis')
    
    Returns:
        PIL Image with heatmap overlay
    """
    w, h = image.size
    attn_up = upscale_map(attn_map, (h, w))
    attn_up = np.clip(attn_up, 0, 1)
    
    # Apply colormap to get RGBA heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = (cmap(attn_up) * 255).astype(np.uint8)
    heatmap_img  = Image.fromarray(heatmap_rgba).convert("RGBA")
    
    # Blend with original image
    image_rgba = image.convert("RGBA")
    blended = Image.blend(image_rgba, heatmap_img, alpha=alpha)
    return blended.convert("RGB")


def visualize_token_maps(
    image: Image.Image,
    token_maps: torch.Tensor,
    tokens: list,
    skip_tokens: set = None,
    cols: int = 4,
    save_path: str = None,
) -> plt.Figure:
    """
    Create a grid visualization: original image + one heatmap per meaningful token.
    
    This produces the main visual output of the project -- for each word in
    the prompt, you see where the model "placed" that concept in the image.
    
    Args:
        image: The generated PIL Image
        token_maps: [seq_len, res, res] tensor of per-token attention maps
        tokens: List of decoded token strings
        skip_tokens: Set of tokens to skip (special tokens, padding)
        cols: Number of columns in the grid
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    if skip_tokens is None:
        skip_tokens = DEFAULT_SKIP_TOKENS
    
    # Filter out special/padding tokens
    display = [
        (i, tok) for i, tok in enumerate(tokens)
        if tok not in skip_tokens and not tok.startswith("<")
    ]
    
    n = len(display) + 1  # +1 for original image
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # First cell: original image
    axes[0].imshow(image)
    axes[0].set_title("Generated image", fontsize=11, pad=6)
    axes[0].axis('off')
    
    # One cell per token
    for plot_idx, (token_idx, token) in enumerate(display, start=1):
        if plot_idx >= len(axes):
            break
        attn = normalize_map(token_maps[token_idx])
        overlaid = overlay_heatmap(image, attn)
        axes[plot_idx].imshow(overlaid)
        axes[plot_idx].set_title(f\\'"{token}"\\', fontsize=11, pad=6)
        axes[plot_idx].axis('off')
    
    # Hide unused axes
    for i in range(len(display) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Cross-attention grounding maps", fontsize=13, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches=\\'tight\\')
    
    return fig
'''

# Fix escaped quotes
content = content.replace("\\\\'", "'")

with open("src/heatmap.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Created src/heatmap.py successfully!")
