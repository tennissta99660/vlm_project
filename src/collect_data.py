"""
Stage 1 — Data Collection
Downloads COCO + Open Images from HuggingFace, generates SDXL synthetic images.
Produces:  data/raw_meta.jsonl
"""
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .utils import append_jsonl, get_logger, load_jsonl, save_checkpoint, load_checkpoint

log = get_logger(__name__)


# ── Synthetic image prompts ───────────────────────────────────────────────────
# 10 categories × varied prompts — generates rich, diverse scenes

SYNTHETIC_PROMPT_TEMPLATES = {
    "indoor_scene": [
        "a cozy living room with bookshelves, plants on windowsill, warm lighting",
        "a busy kitchen with cooking utensils, steam rising, vegetables on counter",
        "a children's bedroom with toys scattered on the floor, colorful walls",
        "a modern office desk setup with multiple monitors, coffee cup, notepads",
    ],
    "outdoor_urban": [
        "a crowded street market in Southeast Asia, colorful stalls, natural daylight",
        "a European cobblestone alley with flower boxes, bicycles parked outside cafes",
        "a rainy night city street with reflections on wet pavement, neon signs",
        "a construction site with workers in hard hats, cranes, scaffolding",
    ],
    "nature": [
        "a dense forest trail with dappled sunlight through trees, mossy rocks",
        "a mountain lake at sunrise with mist, pine trees reflecting in still water",
        "a sandy beach with tide pools, starfish, crabs visible in shallow water",
        "a wheat field during harvest season, combine harvester in the distance",
    ],
    "people_activities": [
        "a scientist in a white lab coat examining a sample under a microscope",
        "an elderly person teaching a young child to play chess, park bench",
        "street musicians performing for a crowd in a busy plaza, evening light",
        "a chef plating a colorful dish in a restaurant kitchen, close up",
    ],
    "animals": [
        "a red fox sitting in autumn leaves, alert expression, forest background",
        "a flock of birds taking off from a wetland at golden hour",
        "a domestic cat and dog sleeping together on a sofa, photorealistic",
        "honeybees on a lavender field, macro photography style",
    ],
    "transportation": [
        "a busy train station platform with commuters and a departing train",
        "an aerial view of a highway interchange at dusk, car light trails",
        "a fishing boat at a small harbor, fishermen unloading the catch",
        "a vintage bicycle leaning against a brick wall, shallow depth of field",
    ],
    "food": [
        "a spread of traditional Indian thali dishes on a banana leaf",
        "a baker decorating a multi-tiered wedding cake, bakery kitchen",
        "a colorful farmers market produce stand with seasonal vegetables",
        "ramen bowl with noodles, soft-boiled egg, broth steaming, close-up",
    ],
    "text_in_scene": [
        "a vintage newspaper stand with headlines visible, city street, 1980s",
        "a coffee shop chalkboard menu with handwritten specials, cozy interior",
        "traffic signs at a busy intersection, crosswalk, pedestrians waiting",
        "a library with visible book spines and shelf labels, warm lighting",
    ],
    "spatial_reasoning": [
        "objects on a table: a red cup to the left of a laptop, book underneath",
        "a cluttered garage with tools hanging on pegboard, boxes stacked high",
        "a supermarket aisle with products on shelves at different heights",
        "a playground with children at different equipment: slide, swings, climbing frame",
    ],
    "counting": [
        "exactly seven apples arranged in a row on a wooden table, natural light",
        "a parking lot with a mix of cars, vans, and motorcycles, aerial view",
        "a classroom with students seated at desks, teacher at whiteboard",
        "a bird's nest with four eggs, close-up, detailed feathers around it",
    ],
}


def collect_coco(cfg: dict, out_dir: str, meta_path: str) -> int:
    """Download COCO images + captions from HuggingFace."""
    log.info("[COCO] Loading dataset...")
    ds = load_dataset("HuggingFaceM4/COCO", split=cfg["data"]["coco_split"], trust_remote_code=True)
    
    ckpt = load_checkpoint(meta_path + ".ckpt")
    saved = 0

    for i, sample in enumerate(tqdm(ds, desc="COCO")):
        item_id = f"coco_{i}"
        if item_id in ckpt:
            continue

        try:
            img: Image.Image = sample["image"].convert("RGB")
            img_path = os.path.join(out_dir, f"{item_id}.jpg")
            img.save(img_path, quality=90)

            # COCO captions can be a list — take the first
            captions = sample.get("sentences", {})
            raw_cap = ""
            if isinstance(captions, dict):
                raw_cap = captions.get("raw", [""])[0]
            elif isinstance(captions, list):
                raw_cap = captions[0] if captions else ""

            record = {
                "id": item_id,
                "source": "coco",
                "image": img_path,
                "width": img.width,
                "height": img.height,
                "raw_caption": raw_cap,
            }
            append_jsonl(record, meta_path)
            save_checkpoint(item_id, meta_path + ".ckpt")
            saved += 1
        except Exception as e:
            log.warning(f"[COCO] Skipping {item_id}: {e}")

    log.info(f"[COCO] Saved {saved} images")
    return saved


def collect_open_images(cfg: dict, out_dir: str, meta_path: str) -> int:
    """Download Open Images subset from HuggingFace (the_cauldron)."""
    log.info("[OpenImages] Loading dataset...")
    # the_cauldron has multiple subsets with question/answer pairs
    ds = load_dataset(
        "HuggingFaceM4/the_cauldron",
        "vsr",  # Visual Spatial Reasoning — great for spatial instruction types
        split=cfg["data"]["open_images_split"],
        trust_remote_code=True
    )

    ckpt = load_checkpoint(meta_path + ".ckpt")
    saved = 0

    for i, sample in enumerate(tqdm(ds, desc="OpenImages/VSR")):
        item_id = f"oi_{i}"
        if item_id in ckpt:
            continue

        try:
            images = sample.get("images", [])
            if not images:
                continue

            img: Image.Image = images[0].convert("RGB")
            img_path = os.path.join(out_dir, f"{item_id}.jpg")
            img.save(img_path, quality=90)

            # Extract existing QA if present (we'll still re-annotate but save this as bonus)
            texts = sample.get("texts", [{}])
            existing_q = texts[0].get("user", "") if texts else ""
            existing_a = texts[0].get("assistant", "") if texts else ""

            record = {
                "id": item_id,
                "source": "open_images_vsr",
                "image": img_path,
                "width": img.width,
                "height": img.height,
                "raw_caption": "",
                "existing_qa": {"q": existing_q, "a": existing_a},
            }
            append_jsonl(record, meta_path)
            save_checkpoint(item_id, meta_path + ".ckpt")
            saved += 1
        except Exception as e:
            log.warning(f"[OpenImages] Skipping {item_id}: {e}")

    log.info(f"[OpenImages] Saved {saved} images")
    return saved


def generate_synthetic(cfg: dict, out_dir: str, meta_path: str) -> int:
    """Generate synthetic images using SDXL."""
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        log.warning("[SDXL] diffusers not installed — skipping synthetic generation.")
        return 0

    log.info("[SDXL] Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()  # saves VRAM

    ckpt = load_checkpoint(meta_path + ".ckpt")
    saved = 0
    n_per_cat = cfg["data"]["synthetic_prompts_per_category"]

    for category, base_prompts in SYNTHETIC_PROMPT_TEMPLATES.items():
        # Expand prompts with style suffixes for more diversity
        style_suffixes = [
            ", photorealistic, 8k",
            ", photography, natural lighting",
            ", documentary photography",
            ", high resolution, sharp",
            ", candid photography",
        ]
        prompts = []
        for bp in base_prompts:
            for suf in style_suffixes:
                prompts.append(bp + suf)
        prompts = prompts[:n_per_cat]

        for j, prompt in enumerate(tqdm(prompts, desc=f"SDXL/{category}")):
            item_id = f"synth_{category}_{j}"
            if item_id in ckpt:
                continue
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, watermark, text overlay, deformed",
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    width=1024,
                    height=768,
                )
                img: Image.Image = result.images[0]
                img_path = os.path.join(out_dir, f"{item_id}.jpg")
                img.save(img_path, quality=88)

                record = {
                    "id": item_id,
                    "source": "sdxl_synthetic",
                    "image": img_path,
                    "width": img.width,
                    "height": img.height,
                    "raw_caption": prompt.split(",")[0],  # first clause as caption
                    "sdxl_prompt": prompt,
                    "category": category,
                }
                append_jsonl(record, meta_path)
                save_checkpoint(item_id, meta_path + ".ckpt")
                saved += 1
            except Exception as e:
                log.warning(f"[SDXL] Skipping {item_id}: {e}")

    log.info(f"[SDXL] Generated {saved} synthetic images")
    return saved


def run(cfg: dict) -> None:
    out_dir  = cfg["paths"]["images"]
    out_path = os.path.join(cfg["paths"]["annotations"], "raw_meta.jsonl")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("=== Stage 1: Data Collection ===")
    n1 = collect_coco(cfg, out_dir, out_path)
    n2 = collect_open_images(cfg, out_dir, out_path)
    n3 = generate_synthetic(cfg, out_dir, out_path)
    total = n1 + n2 + n3
    log.info(f"[Stage 1 done] Total images collected: {total}")
