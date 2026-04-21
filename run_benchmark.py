"""
Multi-prompt benchmark with IoU evaluation.

Usage:
  python run_benchmark.py --out benchmark_results/ --n 8
  python run_benchmark.py --out benchmark_results/ --skip-generation
  python run_benchmark.py --out benchmark_results/ --analysis-only
"""
import argparse
import os
import sys
import gc
import json
import time
import torch
import numpy as np
from PIL import Image

from src.benchmark_prompts import BENCHMARK_PROMPTS, get_categories
from src.attention_extractor import AttentionStore, register_attention_hooks, restore_processors
from src.heatmap import visualize_token_maps
from src.alignment_scorer import load_clip, combined_score
from src.analysis import run_analysis


def get_tokens(prompt: str, tokenizer) -> list:
    ids = tokenizer.encode(prompt)
    return [tokenizer.decode([i]) for i in ids]


def generate_with_attention(pipe, store, prompt, seed, steps=30, guidance=7.5):
    store.reset()
    generator = torch.Generator("cuda").manual_seed(seed)
    hooks = register_attention_hooks(pipe.unet, store)
    with torch.no_grad():
        result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    restore_processors(hooks)
    return result.images[0]


def run_generation_phase(args):
    """Phase 1: Generate images + score with CLIP."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    print(f"\n{'=' * 70}\nPHASE 1: IMAGE GENERATION + CLIP SCORING\n{'=' * 70}")

    print(f"\nLoading Stable Diffusion ({args.model})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()

    print("Loading CLIP model...")
    clip_model, clip_preprocess, clip_tokenizer, clip_device = load_clip("cuda")

    store = AttentionStore()
    total_prompts = len(BENCHMARK_PROMPTS)
    total_images = total_prompts * args.n
    image_count = 0
    start_time = time.time()

    for p_idx, prompt_info in enumerate(BENCHMARK_PROMPTS):
        prompt, category, slug = prompt_info["prompt"], prompt_info["category"], prompt_info["slug"]
        key_tokens = prompt_info["key_tokens"]
        prompt_dir = os.path.join(args.out, category, slug)
        os.makedirs(prompt_dir, exist_ok=True)

        scores_path = os.path.join(prompt_dir, "scores.json")
        if os.path.exists(scores_path) and not args.force:
            print(f"\n[{p_idx+1}/{total_prompts}] SKIP (done): {slug}")
            image_count += args.n
            continue

        print(f"\n{'─' * 70}")
        print(f"[{p_idx+1}/{total_prompts}] {category}/{slug}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"{'─' * 70}")

        tokens = get_tokens(prompt, pipe.tokenizer)
        results = []

        for i in range(args.n):
            seed = 42 + i * 1000
            image_count += 1
            elapsed = time.time() - start_time
            eta = (elapsed / max(image_count, 1)) * (total_images - image_count)
            print(f"  [{i+1}/{args.n}] seed={seed}  (total: {image_count}/{total_images}, ETA: {eta/60:.1f}min)")

            image = generate_with_attention(pipe, store, prompt, seed, args.steps, args.guidance)

            try:
                token_maps = store.get_token_maps(resolution=16)
            except ValueError:
                token_maps = store.get_token_maps(resolution=8)

            scores = combined_score(
                image, prompt, token_maps, tokens,
                clip_model, clip_preprocess, clip_tokenizer, clip_device,
                key_tokens=key_tokens, return_per_token=True,
            )
            print(f"    CLIP={scores['clip_score']:.4f}  Attn={scores['attention_score']:.4f}  Combined={scores['combined']:.4f}")

            img_path = os.path.join(prompt_dir, f"image_{i:02d}_seed{seed}.png")
            image.save(img_path)

            maps_path = os.path.join(prompt_dir, f"token_maps_{i:02d}_seed{seed}.pt")
            torch.save(token_maps, maps_path)

            raw_path = os.path.join(prompt_dir, f"raw_attn_{i:02d}_seed{seed}.pt")
            store.save_raw_storage(raw_path)

            heatmap_path = None
            if i < 2:
                heatmap_path = os.path.join(prompt_dir, f"heatmap_{i:02d}_seed{seed}.png")
                import matplotlib.pyplot as plt
                fig = visualize_token_maps(image, token_maps, tokens, save_path=heatmap_path)
                plt.close(fig)

            results.append({"index": i, "seed": seed, "image_path": img_path,
                            "maps_path": maps_path, "heatmap_path": heatmap_path, "scores": scores})

        results.sort(key=lambda x: x["scores"]["combined"], reverse=True)
        with open(scores_path, "w") as f:
            json.dump({"prompt": prompt, "category": category, "key_tokens": key_tokens, "results": results}, f, indent=2)

        Image.open(results[0]["image_path"]).save(os.path.join(prompt_dir, "BEST.png"))
        print(f"  Best: seed={results[0]['seed']} (combined={results[0]['scores']['combined']:.4f})")

    print(f"\n{'=' * 70}\nPhase 1 complete: {image_count} images in {(time.time()-start_time)/60:.1f} min\n{'=' * 70}")
    del pipe, clip_model
    gc.collect()
    torch.cuda.empty_cache()


def run_iou_phase(args):
    """Phase 2: Evaluate IoU with CLIPSeg."""
    from src.segmentation_eval import CLIPSegEvaluator, evaluate_token_iou

    print(f"\n{'=' * 70}\nPHASE 2: IoU EVALUATION WITH CLIPSeg\n{'=' * 70}")
    evaluator = CLIPSegEvaluator(device="cuda")

    for p_idx, prompt_info in enumerate(BENCHMARK_PROMPTS):
        category, slug = prompt_info["category"], prompt_info["slug"]
        key_tokens, prompt = prompt_info["key_tokens"], prompt_info["prompt"]
        prompt_dir = os.path.join(args.out, category, slug)
        scores_path = os.path.join(prompt_dir, "scores.json")
        iou_path = os.path.join(prompt_dir, "iou_results.json")

        if not os.path.exists(scores_path):
            continue
        if os.path.exists(iou_path) and not args.force:
            print(f"[{p_idx+1}/{len(BENCHMARK_PROMPTS)}] SKIP (done): {slug}")
            continue

        print(f"\n[{p_idx+1}/{len(BENCHMARK_PROMPTS)}] Evaluating IoU: {slug}")

        with open(scores_path) as f:
            scores_data = json.load(f)

        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokens = get_tokens(prompt, tokenizer)

        per_image_iou = []
        for img_result in scores_data["results"]:
            maps_path, img_path = img_result.get("maps_path"), img_result.get("image_path")
            if not maps_path or not os.path.exists(maps_path):
                per_image_iou.append({"mean_iou": 0, "per_token": {}, "seed": img_result["seed"]})
                continue

            token_maps = torch.load(maps_path, map_location="cpu")
            image = Image.open(img_path).convert("RGB")
            iou_result = evaluate_token_iou(image, token_maps, tokens, key_tokens, evaluator)
            iou_result["seed"] = img_result["seed"]
            per_image_iou.append(iou_result)
            print(f"  seed={img_result['seed']}  IoU={iou_result['mean_iou']:.4f}")

        prompt_mean_iou = np.mean([r["mean_iou"] for r in per_image_iou])
        with open(iou_path, "w") as f:
            json.dump({"prompt": prompt, "category": category, "key_tokens": key_tokens,
                        "per_image": per_image_iou, "prompt_mean_iou": round(float(prompt_mean_iou), 4)}, f, indent=2)
        print(f"  Prompt mean IoU: {prompt_mean_iou:.4f}")

    evaluator.unload()
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{'=' * 70}\nPhase 2 complete\n{'=' * 70}")


def main():
    p = argparse.ArgumentParser(description="Cross-attention grounding benchmark")
    p.add_argument("--out", default="benchmark_results/")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_start = time.time()

    print("=" * 70)
    print("CROSS-ATTENTION GROUNDING BENCHMARK")
    print("=" * 70)
    print(f"Prompts: {len(BENCHMARK_PROMPTS)} | Images/prompt: {args.n} | Total: {len(BENCHMARK_PROMPTS) * args.n}")
    print(f"Categories: {', '.join(get_categories())} | Output: {args.out}")

    if not args.skip_generation and not args.analysis_only:
        run_generation_phase(args)
    if not args.analysis_only:
        run_iou_phase(args)
    run_analysis(args.out)

    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE: {(time.time()-total_start)/60:.1f} minutes\n{'=' * 70}")


if __name__ == "__main__":
    main()
