"""
Ablation study: compare paper-grounded attention aggregation strategies.

Usage:
  python run_ablation.py --benchmark-dir benchmark_results/
  python run_ablation.py --benchmark-dir benchmark_results/ --analysis-only
"""
import argparse
import os
import gc
import json
import time
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from src.benchmark_prompts import BENCHMARK_PROMPTS
from src.attention_extractor import AttentionStore
from src.attention_aggregator import AttentionAggregator, AggregationConfig, ABLATION_CONFIGS
from src.segmentation_eval import CLIPSegEvaluator, compute_iou
from src.heatmap import normalize_map, upscale_map

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG_COLORS = {"P1_daam": "#3498db", "P2_p2p": "#2ecc71", "P3_ae": "#e74c3c", "ours": "#9b59b6"}
CONFIG_SHORT = {"P1_daam": "P1: DAAM", "P2_p2p": "P2: P2P", "P3_ae": "P3: A&E", "ours": "Ours"}
CATEGORY_LABELS = {"color_binding": "Color Binding", "spatial": "Spatial", "multi_object": "Multi-Object", "scene": "Scene", "counting": "Counting"}
CATEGORY_ORDER = ["color_binding", "spatial", "multi_object", "scene", "counting"]


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
        'axes.edgecolor': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
        'text.color': '#e0e0e0', 'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
        'grid.color': '#2a2a4a', 'grid.alpha': 0.5, 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 12,
    })

setup_style()


def get_tokens(prompt, tokenizer):
    ids = tokenizer.encode(prompt)
    return [tokenizer.decode([i]) for i in ids]


def evaluate_config_iou(image, token_maps, tokens, eval_tokens, evaluator, attn_threshold_percentile=80.0):
    """Compute IoU for eval_tokens using token_maps."""
    w, h = image.size
    per_token_iou = {}

    for eval_tok in eval_tokens:
        words = eval_tok.lower().split()
        token_idx = None
        for word in words:
            if len(word) <= 2:
                continue
            for i, tok in enumerate(tokens):
                if word in tok.lower().strip():
                    token_idx = i
                    break
            if token_idx is not None:
                break

        if token_idx is None:
            for i, tok in enumerate(tokens):
                if words[0] in tok.lower().strip():
                    token_idx = i
                    break

        if token_idx is None or token_idx >= token_maps.shape[0]:
            continue

        attn = normalize_map(token_maps[token_idx])
        attn_up = np.clip(upscale_map(attn, (h, w)), 0, 1)
        threshold = np.percentile(attn_up, attn_threshold_percentile)
        attn_binary = (attn_up >= threshold).astype(np.float32)
        clipseg_mask = evaluator.get_segmentation_mask(image, eval_tok)
        per_token_iou[eval_tok] = round(compute_iou(attn_binary, clipseg_mask), 4)

    iou_values = list(per_token_iou.values())
    return {"per_token": per_token_iou, "mean_iou": round(float(np.mean(iou_values)), 4) if iou_values else 0.0}


def run_ablation_evaluation(benchmark_dir, force=False):
    """Run all configs on saved benchmark data."""
    ablation_dir = os.path.join(benchmark_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    has_raw = False
    for prompt_info in BENCHMARK_PROMPTS:
        prompt_dir = os.path.join(benchmark_dir, prompt_info["category"], prompt_info["slug"])
        if os.path.isdir(prompt_dir):
            if any(f.startswith("raw_attn_") for f in os.listdir(prompt_dir)
                   if os.path.isfile(os.path.join(prompt_dir, f))):
                has_raw = True
                break

    if not has_raw:
        print("WARNING: No raw_attn_*.pt files. Re-run benchmark with --force.\n")

    evaluator = CLIPSegEvaluator(device="cuda")
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    all_results = {}

    for config_key, config in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"CONFIG: {config.name}")
        print(f"  daam_all_res={config.daam_all_res}  p2p_late_frac={config.p2p_late_timestep_fraction}")
        print(f"  ae_gaussian={config.ae_gaussian_smooth}  compound_eval={config.compound_eval}")
        print(f"{'=' * 70}")

        aggregator = AttentionAggregator(config)
        config_results = []

        for p_idx, prompt_info in enumerate(BENCHMARK_PROMPTS):
            category, slug = prompt_info["category"], prompt_info["slug"]
            prompt, key_tokens = prompt_info["prompt"], prompt_info["key_tokens"]
            compound_tokens = prompt_info.get("compound_tokens", key_tokens)

            prompt_dir = os.path.join(benchmark_dir, category, slug)
            scores_path = os.path.join(prompt_dir, "scores.json")
            if not os.path.exists(scores_path):
                continue

            with open(scores_path) as f:
                scores_data = json.load(f)

            tokens = get_tokens(prompt, tokenizer)
            eval_tokens = compound_tokens if config.compound_eval else key_tokens

            per_image = []
            for img_result in scores_data["results"]:
                seed, idx = img_result["seed"], img_result["index"]
                img_path = img_result.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")
                raw_path = os.path.join(prompt_dir, f"raw_attn_{idx:02d}_seed{seed}.pt")

                if has_raw and os.path.exists(raw_path):
                    raw = AttentionStore.load_raw_storage(raw_path)
                    token_maps = aggregator.aggregate(raw)
                else:
                    maps_path = img_result.get("maps_path", "")
                    if not os.path.exists(maps_path):
                        continue
                    token_maps = torch.load(maps_path, map_location="cpu")

                iou_result = evaluate_config_iou(image, token_maps, tokens, eval_tokens, evaluator)
                iou_result["seed"] = seed
                per_image.append(iou_result)

            if per_image:
                pmean = float(np.mean([r["mean_iou"] for r in per_image]))
                config_results.append({
                    "category": category, "slug": slug,
                    "prompt_mean_iou": round(pmean, 4), "per_image": per_image,
                })
                print(f"  [{p_idx+1}/15] {slug}: IoU={pmean:.4f}")

        overall = round(float(np.mean([r["prompt_mean_iou"] for r in config_results])), 4) if config_results else 0.0
        all_results[config_key] = {"config_name": config.name, "prompts": config_results, "overall_mean_iou": overall}

    evaluator.unload()
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

    path = os.path.join(ablation_dir, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {path}")
    return all_results


def _cat_means(results, config_key):
    out = defaultdict(list)
    for pr in results[config_key].get("prompts", []):
        out[pr["category"]].append(pr["prompt_mean_iou"])
    return {c: np.mean(v) for c, v in out.items()}


def run_ablation_analysis(benchmark_dir, results=None):
    ablation_dir = os.path.join(benchmark_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    if results is None:
        with open(os.path.join(ablation_dir, "ablation_results.json")) as f:
            results = json.load(f)

    print(f"\n{'=' * 70}\nABLATION ANALYSIS\n{'=' * 70}")
    plot_overall_comparison(results, ablation_dir)
    plot_paper_vs_ours(results, ablation_dir)
    plot_category_breakdown(results, ablation_dir)
    plot_radar_chart(results, ablation_dir)
    generate_comparison_table(results, ablation_dir)
    print(f"\nAll plots saved to: {ablation_dir}")


def plot_overall_comparison(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = list(results.keys())
    ious = [results[c]["overall_mean_iou"] for c in configs]
    colors = [CONFIG_COLORS.get(c, "#fff") for c in configs]
    labels = [CONFIG_SHORT.get(c, c) for c in configs]

    bars = ax.bar(labels, ious, color=colors, alpha=0.85, edgecolor='white', linewidth=1, width=0.55)
    for bar, c in zip(bars, configs):
        if c == "ours":
            bar.set_edgecolor('#f1c40f')
            bar.set_linewidth(3)
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{iou:.4f}', ha='center', va='bottom', fontsize=12, color='#e0e0e0', fontweight='bold')

    best_paper = max(configs[:3], key=lambda c: results[c]["overall_mean_iou"])
    ax.axhline(y=results[best_paper]["overall_mean_iou"],
               color=CONFIG_COLORS[best_paper], linestyle='--', alpha=0.5,
               label=f'Best paper: {CONFIG_SHORT[best_paper]}')

    ax.set_ylabel("Mean IoU with CLIPSeg", fontsize=13)
    ax.set_title("Cross-Attention Grounding: Paper Comparison\nDAAM vs Prompt-to-Prompt vs Attend-and-Excite vs Ours", fontsize=14, pad=15)
    ax.legend(loc='upper left', framealpha=0.8, facecolor='#1a1a2e')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(ious) * 1.3 if ious else 1.0)
    plt.tight_layout()
    p = os.path.join(out_dir, "overall_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Overall comparison -> {p}")


def plot_paper_vs_ours(results, out_dir):
    papers = ["P1_daam", "P2_p2p", "P3_ae"]
    ours_iou = results["ours"]["overall_mean_iou"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for ax, pk in zip(axes, papers):
        paper_iou = results[pk]["overall_mean_iou"]
        delta = ours_iou - paper_iou
        pct = (delta / paper_iou * 100) if paper_iou > 0 else 0

        bars = ax.bar([CONFIG_SHORT[pk], "Ours"], [paper_iou, ours_iou],
                      color=[CONFIG_COLORS[pk], CONFIG_COLORS["ours"]],
                      alpha=0.85, edgecolor='white', linewidth=1, width=0.5)
        bars[1].set_edgecolor('#f1c40f')
        bars[1].set_linewidth(2.5)

        for bar, val in zip(bars, [paper_iou, ours_iou]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, color='#e0e0e0', fontweight='bold')

        sign = "+" if delta >= 0 else ""
        color = '#2ecc71' if delta >= 0 else '#e74c3c'
        ax.text(0.5, 0.92, f'{sign}{pct:.1f}%', transform=ax.transAxes, ha='center', fontsize=16,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e', edgecolor=color, alpha=0.9))
        ax.set_title(f'{results[pk]["config_name"]}\nvs Ours', fontsize=12, pad=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(paper_iou, ours_iou) * 1.35)

    axes[0].set_ylabel("Mean IoU", fontsize=13)
    plt.suptitle("Paper-by-Paper Comparison with Ours", fontsize=15, y=1.02, color='#e0e0e0')
    plt.tight_layout()
    p = os.path.join(out_dir, "paper_vs_ours.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Paper vs Ours -> {p}")


def plot_category_breakdown(results, out_dir):
    fig, ax = plt.subplots(figsize=(14, 7))
    configs = list(results.keys())
    cats = CATEGORY_ORDER
    x = np.arange(len(cats))
    width, n = 0.18, len(configs)

    for i, ck in enumerate(configs):
        cm = _cat_means(results, ck)
        vals = [cm.get(c, 0) for c in cats]
        offset = (i - n/2 + 0.5) * width
        b = ax.bar(x + offset, vals, width, label=CONFIG_SHORT[ck],
                   color=CONFIG_COLORS.get(ck, "#fff"), alpha=0.85, edgecolor='white', linewidth=0.5)
        if ck == "ours":
            for bar in b:
                bar.set_edgecolor('#f1c40f')
                bar.set_linewidth(2)

    ax.set_ylabel("Mean IoU", fontsize=13)
    ax.set_title("IoU by Category: All Methods", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=11)
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a2e', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "category_breakdown.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Category breakdown -> {p}")


def plot_radar_chart(results, out_dir):
    cats = CATEGORY_ORDER
    labels = [CATEGORY_LABELS[c] for c in cats]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#16213e')

    for ck in results:
        cm = _cat_means(results, ck)
        values = [cm.get(c, 0) for c in cats] + [cm.get(cats[0], 0)]
        lw = 3 if ck == "ours" else 1.5
        ax.plot(angles, values, 'o-', linewidth=lw, label=CONFIG_SHORT[ck],
                color=CONFIG_COLORS.get(ck, "#fff"), alpha=0.85)
        ax.fill(angles, values, alpha=0.08, color=CONFIG_COLORS.get(ck, "#fff"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Per-Category Radar: All Methods", fontsize=14, pad=25, color='#e0e0e0')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.8, facecolor='#1a1a2e', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "radar_chart.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Radar chart -> {p}")


def generate_comparison_table(results, out_dir):
    configs = list(results.keys())
    cats = CATEGORY_ORDER
    papers = [c for c in configs if c != "ours"]
    best_paper_key = max(papers, key=lambda c: results[c]["overall_mean_iou"])
    ours_iou = results["ours"]["overall_mean_iou"]

    csv_path = os.path.join(out_dir, "comparison_table.csv")
    with open(csv_path, "w") as f:
        f.write("Method," + ",".join(CATEGORY_LABELS[c] for c in cats) + ",Overall,Delta vs Ours\n")
        for ck in configs:
            cm = _cat_means(results, ck)
            name = results[ck]["config_name"]
            vals = [f"{cm.get(c, 0):.4f}" for c in cats]
            ov = results[ck]["overall_mean_iou"]
            delta = ours_iou - ov
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}" if ck != "ours" else "-"
            f.write(f"{name},{','.join(vals)},{ov:.4f},{delta_str}\n")

    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE: Paper Methods vs Ours")
    print(f"{'=' * 100}")
    header = f"{'Method':<35}"
    for c in cats:
        header += f" {CATEGORY_LABELS[c]:>12}"
    header += f" {'Overall':>10} {'vs Ours':>10}"
    print(header)
    print("-" * 100)

    for ck in configs:
        cm = _cat_means(results, ck)
        name = results[ck]["config_name"]
        if ck == "ours":
            name = ">>> " + name + " <<<"
        row = f"{name:<35}"
        for c in cats:
            row += f" {cm.get(c, 0):>12.4f}"
        ov = results[ck]["overall_mean_iou"]
        delta = ours_iou - ov
        sign = "+" if delta >= 0 else ""
        row += f" {ov:>10.4f}"
        row += f"  {sign}{delta:.4f}" if ck != "ours" else f"  {'(ours)':>8}"
        print(row)

    best_iou = results[best_paper_key]["overall_mean_iou"]
    improvement = ours_iou - best_iou
    pct = (improvement / best_iou * 100) if best_iou > 0 else 0
    sign = "+" if improvement >= 0 else ""
    print(f"\nBest paper: {results[best_paper_key]['config_name']} (IoU={best_iou:.4f})")
    print(f"Ours:       IoU={ours_iou:.4f}  ({sign}{improvement:.4f}, {sign}{pct:.1f}%)")
    print(f"\nTable saved -> {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Run ablation study")
    p.add_argument("--benchmark-dir", default="benchmark_results/")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("CROSS-ATTENTION GROUNDING: PAPER COMPARISON + ABLATION")
    print("=" * 70)
    print("\nMethods: P1=DAAM, P2=P2P, P3=A&E, Ours=Combined\n")

    if not args.analysis_only:
        results = run_ablation_evaluation(args.benchmark_dir, args.force)
        run_ablation_analysis(args.benchmark_dir, results)
    else:
        run_ablation_analysis(args.benchmark_dir)

    print(f"\n{'=' * 70}\nABLATION COMPLETE: {(time.time()-t0)/60:.1f} minutes\n{'=' * 70}")


if __name__ == "__main__":
    main()
