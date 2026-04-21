"""
Generate N images, extract attention maps, visualize, score and re-rank.

Usage:
  python generate_and_rank.py --prompt "a red bicycle against a brick wall" --n 4 --out results/
"""
import argparse
import os
import torch
import json
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

from src.attention_extractor import AttentionStore, register_attention_hooks, restore_processors
from src.heatmap import visualize_token_maps
from src.alignment_scorer import load_clip, combined_score


def get_tokens(prompt: str, tokenizer) -> list:
    ids = tokenizer.encode(prompt)
    return [tokenizer.decode([i]) for i in ids]


def generate_with_attention(pipe, store, prompt, seed, num_inference_steps=30, guidance_scale=7.5):
    """Generate a single image while capturing cross-attention maps."""
    store.reset()
    generator = torch.Generator("cuda").manual_seed(seed)
    hooks = register_attention_hooks(pipe.unet, store)

    with torch.no_grad():
        result = pipe(prompt, num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale, generator=generator)

    restore_processors(hooks)
    return result.images[0]


def main():
    p = argparse.ArgumentParser(description="Generate images with cross-attention grounding")
    p.add_argument("--prompt", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", default="results/")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.model}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()

    print("Loading CLIP model...")
    clip_model, clip_preprocess, clip_tokenizer, device = load_clip("cuda")

    tokens = get_tokens(args.prompt, pipe.tokenizer)
    print(f"Prompt tokens: {tokens}")

    store = AttentionStore()
    results = []

    for i in range(args.n):
        seed = 42 + i * 1000
        print(f"\n[{i+1}/{args.n}] Generating with seed {seed}...")

        image = generate_with_attention(pipe, store, args.prompt, seed, args.steps, args.guidance)

        try:
            token_maps = store.get_token_maps(resolution=16)
        except ValueError:
            token_maps = store.get_token_maps(resolution=8)

        scores = combined_score(
            image, args.prompt, token_maps, tokens,
            clip_model, clip_preprocess, clip_tokenizer, device
        )
        print(f"  CLIP: {scores['clip_score']:.4f} | "
              f"Attention: {scores['attention_score']:.4f} | "
              f"Combined: {scores['combined']:.4f}")

        img_path = os.path.join(args.out, f"image_{i:02d}_seed{seed}.png")
        image.save(img_path)

        heatmap_path = os.path.join(args.out, f"heatmap_{i:02d}_seed{seed}.png")
        fig = visualize_token_maps(image, token_maps, tokens, save_path=heatmap_path)
        import matplotlib.pyplot as plt
        plt.close(fig)

        results.append({"index": i, "seed": seed, "image_path": img_path,
                         "heatmap_path": heatmap_path, "scores": scores})

    results.sort(key=lambda x: x["scores"]["combined"], reverse=True)

    print("\n" + "=" * 60)
    print("RANKING (best -> worst):")
    print("=" * 60)
    for rank, r in enumerate(results):
        print(f"  #{rank+1}  seed={r['seed']}  combined={r['scores']['combined']:.4f}  "
              f"(clip={r['scores']['clip_score']:.4f}, attn={r['scores']['attention_score']:.4f})")

    best_dst = os.path.join(args.out, "BEST.png")
    Image.open(results[0]["image_path"]).save(best_dst)
    print(f"\nBest image saved -> {best_dst}")

    scores_path = os.path.join(args.out, "scores.json")
    with open(scores_path, "w") as f:
        json.dump({"prompt": args.prompt, "results": results}, f, indent=2)
    print(f"Scores saved -> {scores_path}")


if __name__ == "__main__":
    main()
