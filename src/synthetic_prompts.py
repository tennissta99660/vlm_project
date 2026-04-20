"""
Advanced synthetic prompt generator.
Uses Llama 3.1 to generate creative, diverse SDXL prompts from seed concepts.
Run standalone: python -m src.synthetic_prompts --n 500 --out data/synth_prompts.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import get_logger, retry

log = get_logger(__name__)

# ── Seed concepts — model expands these into full SDXL prompts ────────────────
SEED_CONCEPTS = [
    # Scenes with strong spatial relationships (great for reasoning + referential)
    "busy intersection with traffic lights, pedestrians, vehicles",
    "laboratory bench with multiple tools and equipment",
    "kitchen counter with food ingredients in preparation",
    "workshop with tools hanging on wall and work in progress",
    "cluttered home office with books, papers, multiple screens",
    "market stall with diverse products on display",
    "children's classroom with decorations, desks, teaching materials",
    "hospital reception area with staff and visitors",
    "library reading room with bookshelves and people studying",
    "outdoor cafe with multiple tables, customers, menu boards",

    # Scenes with text/signs (OCR instruction type)
    "storefront with signage, price tags, window displays",
    "highway junction with road signs and direction indicators",
    "bulletin board with notices, flyers, and printed text",
    "newspaper rack with visible headlines",
    "blackboard with handwritten equations and diagrams",

    # Activities (great for activity + causal types)
    "artist in the middle of painting a canvas, brushes, palette",
    "mechanic working under a car hood in a garage",
    "chef preparing multiple dishes simultaneously in restaurant kitchen",
    "construction workers pouring concrete for a foundation",
    "surgeon performing an operation, medical team around the table",

    # Nature scenes with countable elements
    "tidal pool with starfish, sea urchins, crabs, anemones visible",
    "bird feeder surrounded by multiple species of birds",
    "garden with rows of different vegetables at varying growth stages",
    "coral reef with fish, sea turtles, and other marine life",
    "mountainside with hikers at different elevations on a trail",

    # Comparison opportunities (two distinct regions)
    "before and after renovation of a room, split down the middle",
    "two people with very different workspaces side by side",
    "old building next to a modern glass skyscraper",
    "dry desert landscape adjacent to a lush garden with irrigation",

    # India-specific scenes (relevant to your background)
    "traditional Indian wedding ceremony with decorations and guests",
    "street food vendor in Mumbai with signboard and multiple dishes",
    "Indian classroom with students and teacher, Hindi text on board",
    "colorful festival celebration with lights, crowds, decorations",
]

QUALITY_SUFFIXES = [
    "photorealistic, natural lighting, high resolution",
    "documentary photography style, candid",
    "professional photography, sharp focus",
    "environmental portrait, wide angle",
    "overhead aerial view, detailed",
]

SYSTEM_PROMPT = """You are generating text-to-image prompts for a research dataset.
Given a seed concept, generate 3 diverse, specific image prompts that:
1. Include precise details about objects, their positions, and quantities
2. Describe lighting, time of day, and mood
3. Vary in composition (close-up, wide shot, medium shot)
4. Are suitable for generating photorealistic images with Stable Diffusion

Return ONLY a JSON array of 3 strings, no other text.
Example output: ["prompt 1", "prompt 2", "prompt 3"]"""


@retry(times=2, delay=1.0)
def expand_concept(concept: str, ollama_url: str, model: str) -> list[str]:
    """Use Llama to expand a seed concept into 3 detailed SDXL prompts."""
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": f"<|system|>{SYSTEM_PROMPT}<|user|>Seed concept: {concept}<|assistant|>",
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 300},
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    # Strip markdown fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    prompts = json.loads(raw.strip())
    assert isinstance(prompts, list)

    # Append a quality suffix to each
    return [f"{p}, {random.choice(QUALITY_SUFFIXES)}" for p in prompts]


def generate_prompt_list(n: int, ollama_url: str, model: str) -> list[dict]:
    """Generate n prompts by expanding random seed concepts."""
    all_prompts = []
    seeds = random.choices(SEED_CONCEPTS, k=(n // 3) + 1)

    for seed in tqdm(seeds, desc="Expanding concepts"):
        try:
            expanded = expand_concept(seed, ollama_url, model)
            for p in expanded:
                all_prompts.append({"prompt": p, "seed_concept": seed})
                if len(all_prompts) >= n:
                    break
        except Exception as e:
            log.warning(f"Failed to expand '{seed[:40]}...': {e}")

        if len(all_prompts) >= n:
            break

    return all_prompts[:n]


def main():
    p = argparse.ArgumentParser(description="Generate SDXL prompt list via Llama")
    p.add_argument("--n",          type=int, default=200,     help="Number of prompts to generate")
    p.add_argument("--out",        default="data/synth_prompts.json")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model",      default="llama3.1:8b")
    args = p.parse_args()

    log.info(f"Generating {args.n} prompts via Llama...")
    prompts = generate_prompt_list(args.n, args.ollama_url, args.model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(prompts, f, indent=2)

    log.info(f"Saved {len(prompts)} prompts → {args.out}")
    log.info("Sample:")
    for p in prompts[:3]:
        log.info(f"  {p['prompt'][:100]}...")


if __name__ == "__main__":
    main()
