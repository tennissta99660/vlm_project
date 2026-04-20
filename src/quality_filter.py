"""
Stage 4 — Quality Filtering
Applies three filters to instruction pairs:
  1. CLIP score  — image–response alignment (≥ 0.20)
  2. Hallucination check — response doesn't mention objects absent from annotation
  3. Diversity dedup — cosine similarity deduplication across all instructions
Writes filtered/filtered.jsonl and filtered/rejected.jsonl
"""
import os
from collections import defaultdict

import numpy as np
import open_clip
import spacy
import torch
from PIL import Image
from tqdm import tqdm

from .utils import append_jsonl, get_logger, load_jsonl, save_json

log = get_logger(__name__)


# ── Load models once ──────────────────────────────────────────────────────────

def load_clip(cfg: dict):
    model_name   = cfg["filter"]["clip_model"]
    pretrained   = cfg["filter"]["clip_pretrained"]
    log.info(f"[Filter] Loading CLIP {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, preprocess, tokenizer, device


def load_ner():
    log.info("[Filter] Loading spaCy NER model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        log.warning("spaCy model missing. Run: python -m spacy download en_core_web_sm")
        nlp = None
    return nlp


# ── Filter 1: CLIP score ──────────────────────────────────────────────────────

def compute_clip_score(
    img_path: str,
    text: str,
    model,
    preprocess,
    tokenizer,
    device: str,
) -> float:
    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        # CLIP has a 77-token limit — truncate text safely
        tok = tokenizer([text[:200]])
        with torch.inference_mode():
            img_feat  = model.encode_image(img)
            text_feat = model.encode_text(tok.to(device))
            img_feat  /= img_feat.norm(dim=-1, keepdim=True)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return (img_feat @ text_feat.T).item()
    except Exception as e:
        log.warning(f"[CLIP] score error for {img_path}: {e}")
        return 0.0


# ── Filter 2: Hallucination check ────────────────────────────────────────────

def extract_nouns(text: str, nlp) -> set[str]:
    """Extract meaningful nouns from text using spaCy NER + POS."""
    if nlp is None:
        return set(w.lower() for w in text.split() if len(w) > 3)
    doc = nlp(text.lower())
    nouns = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            nouns.add(token.lemma_)
    for ent in doc.ents:
        nouns.add(ent.text.lower())
    return nouns


def hallucination_score(response: str, annotation: dict, nlp) -> float:
    """
    Returns fraction of response nouns grounded in the annotation.
    Score of 1.0 = fully grounded. Score < 0.5 likely hallucinating.
    """
    # Build a bag-of-words from the annotation
    ann_text = " ".join([
        annotation.get("dense_caption", ""),
        annotation.get("scene", ""),
        " ".join(annotation.get("activities", [])),
        " ".join(annotation.get("relationships", [])),
        " ".join(
            o.get("name", "") for o in annotation.get("objects", [])
            if isinstance(o, dict)
        ),
    ])
    ann_nouns = extract_nouns(ann_text, nlp)
    if not ann_nouns:
        return 1.0  # can't check — pass through

    resp_nouns = extract_nouns(response, nlp)
    if not resp_nouns:
        return 1.0  # response has no extractable nouns — pass through

    # What fraction of response nouns appear in annotation?
    overlap = resp_nouns & ann_nouns
    return len(overlap) / len(resp_nouns)


# ── Filter 3: Diversity deduplication ────────────────────────────────────────

def embed_texts_batch(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text safely
        toks = tokenizer([t[:200] for t in batch])
        with torch.inference_mode():
            emb = model.encode_text(toks.to(device))
            emb /= emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def deduplicate(
    records: list[dict],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Keep only records whose instruction embedding is below threshold
    cosine similarity with all previously kept embeddings."""
    kept, rejected = [], []
    kept_embs: list[np.ndarray] = []

    for rec, emb in zip(records, embeddings):
        if not kept_embs:
            kept.append(rec); kept_embs.append(emb)
            continue
        sims = np.array(kept_embs) @ emb  # (N,)
        if sims.max() < threshold:
            kept.append(rec); kept_embs.append(emb)
        else:
            rejected.append(rec)

    return kept, rejected


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    in_path      = os.path.join(cfg["paths"]["instructions"], "instructions.jsonl")
    out_kept     = os.path.join(cfg["paths"]["filtered"], "filtered.jsonl")
    out_rejected = os.path.join(cfg["paths"]["filtered"], "rejected.jsonl")
    stats_path   = os.path.join(cfg["paths"]["filtered"], "filter_stats.json")

    records = load_jsonl(in_path)
    log.info(f"[Stage 4] Filtering {len(records)} instruction pairs...")

    # Load models
    clip_model, preprocess, tokenizer, device = load_clip(cfg)
    nlp = load_ner() if cfg["filter"]["hallucination_check"] else None

    min_clip     = cfg["filter"]["min_clip_score"]
    min_words    = cfg["filter"]["min_response_words"]
    max_words    = cfg["filter"]["max_response_words"]
    dedup_thresh = cfg["filter"]["dedup_threshold"]

    # ── Pass 1: individual record filters ─────────────────────────────────────
    passed_p1   = []
    reject_reasons = defaultdict(int)

    for rec in tqdm(records, desc="Pass 1 (CLIP + halu)"):
        resp  = rec["response"]
        words = len(resp.split())

        # Length check
        if words < min_words:
            reject_reasons["too_short"] += 1; continue
        if words > max_words:
            reject_reasons["too_long"] += 1; continue

        # Instruction sanity
        if len(rec["instruction"].strip()) < 10:
            reject_reasons["bad_instruction"] += 1; continue

        # CLIP score
        clip_s = compute_clip_score(rec["image"], resp, clip_model, preprocess, tokenizer, device)
        if clip_s < min_clip:
            reject_reasons["low_clip_score"] += 1
            append_jsonl({**rec, "reject_reason": "low_clip_score", "clip_score": clip_s}, out_rejected)
            continue

        # Hallucination check
        if nlp is not None:
            h_score = hallucination_score(resp, rec.get("annotation", {}), nlp)
            if h_score < 0.30:   # less than 30% of response nouns grounded
                reject_reasons["hallucination"] += 1
                append_jsonl({**rec, "reject_reason": "hallucination", "h_score": h_score}, out_rejected)
                continue
            rec["h_score"] = round(h_score, 3)

        rec["clip_score"] = round(clip_s, 3)
        passed_p1.append(rec)

    log.info(f"  After Pass 1: {len(passed_p1)}/{len(records)} passed. Rejected: {dict(reject_reasons)}")

    # ── Pass 2: type-balanced diversity dedup ─────────────────────────────────
    # Deduplicate within each instruction type to ensure type diversity
    passed_p2 = []
    by_type: dict[str, list] = defaultdict(list)
    for rec in passed_p1:
        by_type[rec["type"]].append(rec)

    for itype, type_records in by_type.items():
        instructions = [r["instruction"] for r in type_records]
        embs = embed_texts_batch(instructions, clip_model, tokenizer, device)
        kept, rej = deduplicate(type_records, embs, dedup_thresh)
        passed_p2.extend(kept)
        for r in rej:
            append_jsonl({**r, "reject_reason": "duplicate"}, out_rejected)
        log.info(f"  [{itype}] {len(kept)}/{len(type_records)} after dedup")

    # Save kept records
    for rec in passed_p2:
        # Strip heavy annotation field before saving (save space)
        out_rec = {k: v for k, v in rec.items() if k != "annotation"}
        append_jsonl(out_rec, out_kept)

    # ── Stats ──────────────────────────────────────────────────────────────────
    type_counts = defaultdict(int)
    for rec in passed_p2:
        type_counts[rec["type"]] += 1

    stats = {
        "input":         len(records),
        "after_pass1":   len(passed_p1),
        "after_dedup":   len(passed_p2),
        "rejection_reasons": dict(reject_reasons),
        "by_type":       dict(type_counts),
        "avg_clip_score": round(
            float(np.mean([r.get("clip_score", 0) for r in passed_p2])), 3
        ) if passed_p2 else 0,
    }
    save_json(stats, stats_path)

    log.info(f"[Stage 4 done] Kept {len(passed_p2)} samples. Stats → {stats_path}")
    log.info(f"  By type: {dict(type_counts)}")
