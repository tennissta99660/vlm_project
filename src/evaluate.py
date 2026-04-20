"""
Stage 6 — Evaluation & Ablation
Runs ablation study and formats results for the research paper/report.
Also provides helper to invoke lmms-eval benchmarks.
"""
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_logger, load_jsonl, save_json

log = get_logger(__name__)


# ── lmms-eval runner ──────────────────────────────────────────────────────────

def run_lmms_eval(
    model_path: str,
    tasks: list[str],
    output_dir: str,
    model_type: str = "llava",
    num_gpus: int = 1,
) -> dict:
    """
    Runs lmms-eval and returns parsed results.
    model_path: path to merged model OR HuggingFace repo ID
    tasks: e.g. ["mmbench_en", "pope", "mme", "seedbench"]
    """
    tasks_str = ",".join(tasks)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "accelerate.commands.launch",
        f"--num_processes={num_gpus}",
        "-m", "lmms_eval",
        "--model", model_type,
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks_str,
        "--batch_size", "1",
        "--output_path", str(out_path),
        "--log_samples",
    ]

    log.info(f"[Eval] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"[Eval] lmms-eval failed:\n{result.stderr}")
        return {}

    # Parse results JSON written by lmms-eval
    results_file = out_path / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


# ── Ablation study ────────────────────────────────────────────────────────────

def ablation_by_type(
    filtered_path: str,
    eval_fn,
    output_dir: str,
) -> pd.DataFrame:
    """
    Train a separate LoRA on each instruction type subset and compare benchmarks.
    eval_fn: callable(model_path) -> dict of metric scores
    Returns a DataFrame with rows=types, cols=metrics.
    """
    records = load_jsonl(filtered_path)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)

    rows = []
    for itype, subset in by_type.items():
        log.info(f"[Ablation] Type '{itype}': {len(subset)} samples")
        # In practice, you'd fine-tune a model on `subset` here and evaluate it.
        # This function is a scaffold — plug in your fine-tuning step.
        row = {
            "type":   itype,
            "n":      len(subset),
            "avg_clip_score": round(np.mean([r.get("clip_score", 0) for r in subset]), 3),
            "avg_response_len": round(np.mean([len(r["response"].split()) for r in subset]), 1),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    csv_path = os.path.join(output_dir, "ablation_by_type.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"[Ablation] Saved → {csv_path}")
    return df


def dataset_statistics(filtered_path: str, output_dir: str) -> dict:
    """Compute and print dataset statistics for the paper."""
    records = load_jsonl(filtered_path)

    by_type   = defaultdict(int)
    by_source = defaultdict(int)
    clip_scores = []
    resp_lens   = []

    for r in records:
        by_type[r.get("type",   "unknown")] += 1
        by_source[r.get("source","unknown")] += 1
        clip_scores.append(r.get("clip_score", 0.0))
        resp_lens.append(len(r["response"].split()))

    stats = {
        "total": len(records),
        "by_type":   dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])),
        "clip_score": {
            "mean": round(float(np.mean(clip_scores)), 3),
            "std":  round(float(np.std(clip_scores)), 3),
            "min":  round(float(np.min(clip_scores)), 3),
            "max":  round(float(np.max(clip_scores)), 3),
        },
        "response_length_words": {
            "mean": round(float(np.mean(resp_lens)), 1),
            "std":  round(float(np.std(resp_lens)), 1),
            "min":  int(np.min(resp_lens)),
            "max":  int(np.max(resp_lens)),
        },
    }

    save_json(stats, os.path.join(output_dir, "dataset_stats.json"))
    return stats


def comparison_table(results: dict) -> str:
    """
    Format a benchmark comparison table for the paper.
    results = {
        "baseline":  {"MMBench": 68.1, "POPE_acc": 85.2},
        "ours":      {"MMBench": 71.4, "POPE_acc": 87.8},
        "llava665k": {"MMBench": 72.0, "POPE_acc": 88.5},
    }
    """
    all_metrics = sorted({m for r in results.values() for m in r})
    header = f"{'Model':<20}" + "".join(f"{m:<14}" for m in all_metrics)
    sep    = "-" * len(header)
    rows   = [header, sep]

    for model_name, scores in results.items():
        row = f"{model_name:<20}"
        for m in all_metrics:
            val = scores.get(m, "-")
            row += f"{str(val):<14}"
        rows.append(row)

    table = "\n".join(rows)
    log.info(f"\n{table}\n")
    return table


def run(cfg: dict) -> None:
    log.info("=== Stage 6: Evaluation ===")

    filtered_path = os.path.join(cfg["paths"]["filtered"], "filtered.jsonl")
    output_dir    = "results"
    Path(output_dir).mkdir(exist_ok=True)

    # 1. Dataset stats
    log.info("[Eval] Computing dataset statistics...")
    stats = dataset_statistics(filtered_path, output_dir)
    log.info(f"  Total samples: {stats['total']}")
    log.info(f"  By type: {stats['by_type']}")
    log.info(f"  CLIP score: {stats['clip_score']}")

    # 2. Ablation by instruction type
    log.info("[Eval] Running ablation study...")
    ablation_by_type(filtered_path, eval_fn=None, output_dir=output_dir)

    # 3. Benchmark eval — point at your fine-tuned merged model
    merged_model = os.path.join(cfg["training"]["output_dir"], "merged")
    if os.path.exists(merged_model):
        log.info(f"[Eval] Running lmms-eval on {merged_model}...")
        run_lmms_eval(
            model_path=merged_model,
            tasks=["mmbench_en", "pope", "mme"],
            output_dir=os.path.join(output_dir, "lmms_eval"),
        )
    else:
        log.info(f"[Eval] No merged model found at {merged_model}. Run fine-tuning first.")
        log.info("       Then merge LoRA: llamafactory-cli export configs/train_lora.yaml")

    log.info("[Stage 6 done] Results in ./results/")
