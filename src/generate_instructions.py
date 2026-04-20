"""
Stage 3 — Instruction Generation with Llama 3.1 (via Ollama)
Reads annotated_meta.jsonl, generates N instruction-response pairs per image.
Writes instructions.jsonl
"""
import json
import os
import random
from typing import Optional

import requests
from tqdm import tqdm

from .utils import (
    append_jsonl, get_logger,
    load_checkpoint, load_jsonl,
    retry, save_checkpoint
)

log = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are building a high-quality visual instruction tuning dataset for training vision-language models.
Given a structured annotation of an image, generate one instruction-response pair.
Your output must be ONLY a valid JSON object with exactly two keys: "instruction" and "response".
No markdown, no explanation, no preamble. Only JSON.
Rules:
- The instruction must be answerable from the image alone.
- The response must be factual, specific, and grounded in the annotation.
- Never mention the word "annotation" or "JSON" in instruction or response.
- Write naturally, as if a human is asking about a real photo."""

# ── Per-type prompting templates ──────────────────────────────────────────────
TYPE_PROMPTS = {
    "descriptive": {
        "task": "Write one open-ended question asking for a detailed description of the scene, a specific object, or the overall image.",
        "response_hint": "Provide a detailed, specific description (2–4 sentences). Reference colors, textures, positions.",
        "condition": None,  # always eligible
    },
    "counting": {
        "task": "Write one question that requires counting a specific category of object visible in the image.",
        "response_hint": "Give the exact count and briefly describe where the objects are located.",
        "condition": lambda ann: any(
            isinstance(o, dict) and o.get("count", 1) >= 1
            for o in ann.get("objects", [])
        ),
    },
    "reasoning": {
        "task": "Write one question requiring multi-step visual reasoning (e.g. infer purpose, cause, sequence, or consequence from what is visible).",
        "response_hint": "Think step-by-step. Show the reasoning chain explicitly before giving the final answer. Response should be 3–5 sentences.",
        "condition": lambda ann: len(ann.get("relationships", [])) > 0 or len(ann.get("activities", [])) > 0,
    },
    "comparison": {
        "task": "Write one question comparing two distinct objects, regions, or attributes visible in the image.",
        "response_hint": "Compare along at least two dimensions (e.g. size, color, position, state). Be specific.",
        "condition": lambda ann: len(ann.get("objects", [])) >= 2,
    },
    "referential": {
        "task": "Write one question using a referring expression to uniquely identify an object (e.g. 'the red object near the window', 'the person on the left').",
        "response_hint": "Identify the object precisely and describe it in detail.",
        "condition": lambda ann: len(ann.get("objects", [])) >= 2,
    },
    "yes_no": {
        "task": "Write one yes/no question about something concrete in the image. Make sure the answer is clearly yes OR clearly no based on the annotation.",
        "response_hint": "Answer with 'Yes' or 'No', then explain briefly (1–2 sentences) referencing specific visual evidence.",
        "condition": None,
    },
    "causal": {
        "task": "Write one question asking WHY something in the image looks the way it does, or what caused a visible state or arrangement.",
        "response_hint": "Provide a plausible causal explanation grounded in visible evidence (2–3 sentences).",
        "condition": lambda ann: len(ann.get("activities", [])) > 0 or ann.get("scene") not in ("abstract", "unknown"),
    },
    "ocr": {
        "task": "Write one question about the text, signs, or labels visible in the image.",
        "response_hint": "Transcribe or describe the text accurately. If multiple text items, list them.",
        "condition": lambda ann: len(ann.get("text_in_image", [])) > 0,
    },
}


def build_prompt(annotation: dict, instruction_type: str) -> str:
    """Build the user message for Llama."""
    tinfo = TYPE_PROMPTS[instruction_type]
    
    # Slim the annotation to reduce token count
    ann_slim = {
        "objects":       annotation.get("objects", [])[:8],  # cap at 8 objects
        "scene":         annotation.get("scene", ""),
        "activities":    annotation.get("activities", []),
        "relationships": annotation.get("relationships", []),
        "text_in_image": annotation.get("text_in_image", []),
        "dense_caption": annotation.get("dense_caption", ""),
    }

    return f"""Image annotation:
{json.dumps(ann_slim, indent=2)}

Task: {tinfo['task']}
Response hint: {tinfo['response_hint']}
Type label: {instruction_type}

Return JSON: {{"instruction": "...", "response": "..."}}"""


@retry(times=3, delay=2.0)
def call_ollama(prompt: str, cfg: dict) -> Optional[dict]:
    """Call local Ollama instance."""
    url   = cfg["instruction"]["ollama_url"].rstrip("/") + "/api/generate"
    model = cfg["instruction"]["ollama_model"]

    full_prompt = f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    resp = requests.post(url, json={
        "model":  model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": cfg["instruction"]["temperature"],
            "num_predict": cfg["instruction"]["max_tokens"],
            "stop": ["<|eot_id|>"],
        }
    }, timeout=60)
    resp.raise_for_status()

    raw = resp.json()["response"].strip()

    # Strip markdown fences if model adds them
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw.strip())
    assert "instruction" in parsed and "response" in parsed, "Missing keys"
    return parsed


def eligible_types(annotation: dict, all_types: list[str]) -> list[str]:
    """Return only instruction types whose conditions are met."""
    eligible = []
    for t in all_types:
        cond = TYPE_PROMPTS[t].get("condition")
        if cond is None or cond(annotation):
            eligible.append(t)
    return eligible or ["descriptive"]  # fallback


def run(cfg: dict) -> None:
    in_path  = os.path.join(cfg["paths"]["annotations"], "annotated_meta.jsonl")
    out_path = os.path.join(cfg["paths"]["instructions"], "instructions.jsonl")
    ckpt     = load_checkpoint(out_path + ".ckpt")

    records = load_jsonl(in_path)
    k       = cfg["instruction"]["instructions_per_image"]
    all_types = cfg["instruction"]["types"]

    log.info(f"[Stage 3] Generating {k} instructions per image × {len(records)} images...")

    # Quick connectivity check
    try:
        requests.get(cfg["instruction"]["ollama_url"], timeout=5)
    except Exception:
        log.error("[Ollama] Cannot connect. Is Ollama running? Try: ollama serve &")
        raise SystemExit(1)

    total_saved = 0

    for rec in tqdm(records, desc="Instruction gen"):
        item_id   = rec["id"]
        ckpt_key  = item_id
        if ckpt_key in ckpt:
            continue

        annotation = rec.get("annotation", {})
        
        # For synthetic images with known category, bias toward richer types
        is_synthetic = rec.get("source") == "sdxl_synthetic"

        elig = eligible_types(annotation, all_types)
        sampled = random.sample(elig, min(k, len(elig)))

        for itype in sampled:
            prompt = build_prompt(annotation, itype)
            pair   = None

            try:
                pair = call_ollama(prompt, cfg)
            except json.JSONDecodeError:
                log.warning(f"[InstrGen] JSON error for {item_id}/{itype}")
            except Exception as e:
                log.warning(f"[InstrGen] Error for {item_id}/{itype}: {e}")

            if pair is None:
                continue

            # Minimal length guard before saving
            resp_words = len(pair["response"].split())
            if resp_words < 8 or len(pair["instruction"].strip()) < 10:
                continue

            out_record = {
                "id":           f"{item_id}_{itype}",
                "image_id":     item_id,
                "image":        rec["image"],
                "source":       rec.get("source", "unknown"),
                "instruction":  pair["instruction"],
                "response":     pair["response"],
                "type":         itype,
                "annotation":   annotation,
                "is_synthetic": is_synthetic,
            }
            append_jsonl(out_record, out_path)
            total_saved += 1

        save_checkpoint(ckpt_key, out_path + ".ckpt")

    log.info(f"[Stage 3 done] Generated {total_saved} instruction-response pairs.")
