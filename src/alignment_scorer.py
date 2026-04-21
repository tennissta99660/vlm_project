"""CLIP + attention-based image-prompt alignment scoring and re-ranking."""
import torch
import numpy as np
from PIL import Image
import open_clip
from .heatmap import normalize_map, upscale_map


def load_clip(device: str = "cuda"):
    """Load OpenCLIP ViT-B-32 model. Returns (model, preprocess, tokenizer, device)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def clip_score(image: Image.Image, text: str, model, preprocess, tokenizer, device) -> float:
    """CLIP cosine similarity between image and text."""
    img_t = preprocess(image).unsqueeze(0).to(device)
    tok = tokenizer([text]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img_t)
        text_feat = model.encode_text(tok)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return (img_feat @ text_feat.T).item()


def _crop_to_attention_region(image: Image.Image, token_maps: torch.Tensor, token_idx: int) -> Image.Image:
    """Crop image to the high-attention region (top 20%) for a token."""
    w, h = image.size
    attn = normalize_map(token_maps[token_idx])
    attn_up = upscale_map(attn, (h, w))

    threshold = np.percentile(attn_up, 80)
    mask = attn_up >= threshold

    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any() or not cols_any.any():
        return None

    rmin, rmax = np.where(rows_any)[0][[0, -1]]
    cmin, cmax = np.where(cols_any)[0][[0, -1]]

    pad_r = max(8, int((rmax - rmin) * 0.1))
    pad_c = max(8, int((cmax - cmin) * 0.1))
    rmin, rmax = max(0, rmin - pad_r), min(h, rmax + pad_r)
    cmin, cmax = max(0, cmin - pad_c), min(w, cmax + pad_c)

    crop = image.crop((cmin, rmin, cmax, rmax))
    if crop.width < 10 or crop.height < 10:
        return None
    return crop


def attention_alignment_score(
    image, token_maps, tokens, model, preprocess, tokenizer, device,
    top_k_tokens=5, key_tokens=None, return_per_token=False,
):
    """Per-token attention-weighted CLIP score.
    Crops high-attention regions and checks CLIP similarity for each token."""
    scores = []
    per_token_scores = {}

    if key_tokens is not None:
        meaningful = []
        for key_tok in key_tokens:
            for i, tok in enumerate(tokens):
                if key_tok.lower() in tok.lower().strip():
                    meaningful.append((i, key_tok))
                    break
    else:
        stop_words = {"the", "a", "an", "of", "in", "on", "at", "and", "or"}
        meaningful = [
            (i, tok) for i, tok in enumerate(tokens)
            if len(tok) > 2 and not tok.startswith("<") and tok not in stop_words
        ][:top_k_tokens]

    for token_idx, token in meaningful:
        if token_idx >= token_maps.shape[0]:
            continue
        crop = _crop_to_attention_region(image, token_maps, token_idx)
        if crop is None:
            continue
        s = clip_score(crop, token, model, preprocess, tokenizer, device)
        scores.append(s)
        per_token_scores[token] = round(s, 4)

    mean_score = float(np.mean(scores)) if scores else 0.0

    if return_per_token:
        return {"per_token": per_token_scores, "mean": round(mean_score, 4), "num_evaluated": len(scores)}
    return mean_score


def combined_score(
    image, prompt, token_maps, tokens, model, preprocess, tokenizer, device,
    w_clip=0.5, w_attn=0.5, key_tokens=None, return_per_token=False,
) -> dict:
    """Combined score = w_clip * CLIP + w_attn * attention alignment."""
    global_s = clip_score(image, prompt, model, preprocess, tokenizer, device)

    if return_per_token:
        attn_result = attention_alignment_score(
            image, token_maps, tokens, model, preprocess, tokenizer, device,
            key_tokens=key_tokens, return_per_token=True,
        )
        attn_s = attn_result["mean"]
    else:
        attn_s = attention_alignment_score(
            image, token_maps, tokens, model, preprocess, tokenizer, device,
            key_tokens=key_tokens,
        )

    result = {
        "clip_score": round(global_s, 4),
        "attention_score": round(attn_s, 4),
        "combined": round(w_clip * global_s + w_attn * attn_s, 4),
    }
    if return_per_token:
        result["per_token_attn"] = attn_result["per_token"]
    return result
