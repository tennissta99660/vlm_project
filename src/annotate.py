"""
Stage 2 — Auto-Annotation with Qwen2-VL-7B-Instruct
Reads raw_meta.jsonl, annotates each image, writes annotated_meta.jsonl
"""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

from .utils import (
    append_jsonl, get_logger,
    load_checkpoint, load_jsonl,
    save_checkpoint, retry
)

log = get_logger(__name__)

# ── Annotation prompt ─────────────────────────────────────────────────────────
ANNOTATION_PROMPT = """Analyze this image carefully and return a JSON object.
Your response must be ONLY valid JSON — no markdown, no explanation.

Return this exact schema:
{
  "objects": [
    {"name": "object name", "location": "where in image (e.g. top-left, center)", "count": 1}
  ],
  "scene": "one of: indoor | outdoor-urban | outdoor-nature | abstract | diagram | text-heavy",
  "lighting": "one of: bright | dim | artificial | natural | mixed",
  "activities": ["list of actions/activities visible"],
  "text_in_image": ["any readable text or signs visible (empty list if none)"],
  "relationships": ["spatial/semantic relationships, e.g. 'cat sits on chair'"],
  "dominant_colors": ["top 3 dominant colors"],
  "dense_caption": "A detailed 3-sentence description of the image covering what is shown, what is happening, and the overall mood or context."
}"""


def load_qwen2vl(cfg: dict):
    """Load Qwen2-VL with optional 4-bit quantisation."""
    model_id = cfg["annotation"]["model_id"]
    use_4bit = cfg["annotation"].get("use_4bit", True)

    log.info(f"[Annotator] Loading {model_id} (4-bit={use_4bit})...")

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    log.info("[Annotator] Model loaded.")
    return model, processor


@retry(times=2, delay=1.0)
def annotate_one(image_path: str, model, processor, cfg: dict) -> dict | None:
    """Annotate a single image. Returns parsed dict or None on failure."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        log.error("qwen-vl-utils not installed. Run: pip install qwen-vl-utils")
        raise

    # Build conversation message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text",  "text": ANNOTATION_PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["annotation"]["max_new_tokens"],
            temperature=cfg["annotation"]["temperature"],
            do_sample=cfg["annotation"]["temperature"] > 0,
        )

    # Decode only generated tokens (strip input tokens)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Parse JSON — strip any accidental markdown fences
    clean = raw_text
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())


def run(cfg: dict) -> None:
    in_path  = os.path.join(cfg["paths"]["annotations"], "raw_meta.jsonl")
    out_path = os.path.join(cfg["paths"]["annotations"], "annotated_meta.jsonl")
    ckpt     = load_checkpoint(out_path + ".ckpt")

    records = load_jsonl(in_path)
    log.info(f"[Stage 2] Annotating {len(records)} images ({len(ckpt)} already done)...")

    model, processor = load_qwen2vl(cfg)
    save_every = cfg["annotation"].get("save_every", 500)
    success = 0

    for i, rec in enumerate(tqdm(records, desc="Annotating")):
        item_id = rec["id"]
        if item_id in ckpt:
            continue

        img_path = rec["image"]
        if not os.path.exists(img_path):
            log.warning(f"Missing image: {img_path} — skipping.")
            continue

        # Basic image validation
        try:
            img = Image.open(img_path)
            if img.width < 32 or img.height < 32:
                log.warning(f"Image too small: {img_path}")
                continue
        except Exception:
            continue

        annotation = None
        try:
            annotation = annotate_one(img_path, model, processor, cfg)
        except json.JSONDecodeError:
            log.warning(f"[Annotate] JSON parse failed for {item_id}")
        except Exception as e:
            log.warning(f"[Annotate] Error on {item_id}: {e}")

        if annotation is None:
            # Save a minimal stub so we don't retry indefinitely
            annotation = {
                "objects": [],
                "scene": "unknown",
                "lighting": "unknown",
                "activities": [],
                "text_in_image": [],
                "relationships": [],
                "dominant_colors": [],
                "dense_caption": rec.get("raw_caption", ""),
            }

        out_rec = {**rec, "annotation": annotation}
        append_jsonl(out_rec, out_path)
        save_checkpoint(item_id, out_path + ".ckpt")
        success += 1

        if (i + 1) % save_every == 0:
            log.info(f"  Progress: {i+1}/{len(records)} — {success} successful annotations")

    log.info(f"[Stage 2 done] Annotated {success}/{len(records)} images.")
