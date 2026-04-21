"""CLIPSeg-based ground-truth evaluation for cross-attention maps."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from .heatmap import normalize_map, upscale_map


class CLIPSegEvaluator:
    """Produces CLIPSeg segmentation masks and computes IoU against attention maps."""

    def __init__(self, device: str = "cuda", model_name: str = "CIDAS/clipseg-rd64-refined"):
        print(f"Loading CLIPSeg ({model_name})...")
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.model = self.model.to(device).eval()
        self.device = device

    def get_segmentation_mask(self, image: Image.Image, text_query: str, threshold: float = 0.5) -> np.ndarray:
        """Binary segmentation mask for a text query. Returns {0,1} array at image resolution."""
        inputs = self.processor(text=[text_query], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.squeeze(0)
        w, h = image.size
        logits_resized = F.interpolate(
            logits.unsqueeze(0).unsqueeze(0), size=(h, w),
            mode='bilinear', align_corners=False,
        ).squeeze()

        prob_map = torch.sigmoid(logits_resized).cpu().numpy()
        return (prob_map >= threshold).astype(np.float32)

    def get_probability_map(self, image: Image.Image, text_query: str) -> np.ndarray:
        """Soft probability map (no thresholding). Returns [0,1] array."""
        inputs = self.processor(text=[text_query], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.squeeze(0)
        w, h = image.size
        logits_resized = F.interpolate(
            logits.unsqueeze(0).unsqueeze(0), size=(h, w),
            mode='bilinear', align_corners=False,
        ).squeeze()

        return torch.sigmoid(logits_resized).cpu().numpy()

    def unload(self):
        self.model = self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("CLIPSeg unloaded from GPU.")


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks. Returns 0 if both are empty."""
    intersection = np.logical_and(mask_a > 0.5, mask_b > 0.5).sum()
    union = np.logical_or(mask_a > 0.5, mask_b > 0.5).sum()
    return float(intersection / union) if union > 0 else 0.0


def evaluate_token_iou(
    image, token_maps, tokens, key_tokens, evaluator,
    attn_threshold_percentile=80.0, clipseg_threshold=0.5,
) -> dict:
    """Compute IoU between attention maps and CLIPSeg masks for each key token."""
    w, h = image.size
    per_token_iou = {}

    for key_tok in key_tokens:
        token_idx = None
        for i, tok in enumerate(tokens):
            if key_tok.lower() in tok.lower().strip():
                token_idx = i
                break

        if token_idx is None or token_idx >= token_maps.shape[0]:
            continue

        attn = normalize_map(token_maps[token_idx])
        attn_up = upscale_map(attn, (h, w))
        attn_up = np.clip(attn_up, 0, 1)

        threshold = np.percentile(attn_up, attn_threshold_percentile)
        attn_binary = (attn_up >= threshold).astype(np.float32)

        clipseg_mask = evaluator.get_segmentation_mask(image, key_tok, threshold=clipseg_threshold)
        iou = compute_iou(attn_binary, clipseg_mask)
        per_token_iou[key_tok] = round(iou, 4)

    iou_values = list(per_token_iou.values())
    return {
        "per_token": per_token_iou,
        "mean_iou": round(float(np.mean(iou_values)), 4) if iou_values else 0.0,
        "num_evaluated": len(iou_values),
    }
