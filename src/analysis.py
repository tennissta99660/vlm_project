"""Aggregate analysis and visualization for benchmark results."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0', 'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a', 'grid.alpha': 0.5, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
})

CATEGORY_LABELS = {"color_binding": "Color Binding", "spatial": "Spatial", "multi_object": "Multi-Object", "scene": "Scene", "counting": "Counting"}
CATEGORY_COLORS = {"color_binding": "#ff6b6b", "spatial": "#4ecdc4", "multi_object": "#45b7d1", "scene": "#96ceb4", "counting": "#ffeaa7"}


def load_benchmark_results(results_dir: str) -> list:
    all_results = []
    for category in os.listdir(results_dir):
        category_dir = os.path.join(results_dir, category)
        if not os.path.isdir(category_dir) or category == "analysis":
            continue
        for slug in os.listdir(category_dir):
            prompt_dir = os.path.join(category_dir, slug)
            if not os.path.isdir(prompt_dir):
                continue
            scores_path = os.path.join(prompt_dir, "scores.json")
            iou_path = os.path.join(prompt_dir, "iou_results.json")
            if not os.path.exists(scores_path):
                continue
            with open(scores_path) as f:
                scores_data = json.load(f)
            iou_data = None
            if os.path.exists(iou_path):
                with open(iou_path) as f:
                    iou_data = json.load(f)
            all_results.append({
                "category": category, "slug": slug,
                "prompt": scores_data.get("prompt", ""),
                "scores": scores_data.get("results", []),
                "iou": iou_data,
            })
    return all_results


def generate_summary_table(results, output_dir):
    category_stats = defaultdict(lambda: {
        "clip_scores": [], "attn_scores": [], "combined_scores": [],
        "iou_scores": [], "num_prompts": 0, "num_images": 0,
    })

    for r in results:
        cat = r["category"]
        category_stats[cat]["num_prompts"] += 1
        for img_result in r["scores"]:
            scores = img_result["scores"]
            category_stats[cat]["clip_scores"].append(scores["clip_score"])
            category_stats[cat]["attn_scores"].append(scores["attention_score"])
            category_stats[cat]["combined_scores"].append(scores["combined"])
            category_stats[cat]["num_images"] += 1
        if r["iou"]:
            for img_iou in r["iou"].get("per_image", []):
                if "mean_iou" in img_iou:
                    category_stats[cat]["iou_scores"].append(img_iou["mean_iou"])

    summary = {}
    for cat, stats in sorted(category_stats.items()):
        summary[cat] = {
            "label": CATEGORY_LABELS.get(cat, cat),
            "num_prompts": stats["num_prompts"],
            "num_images": stats["num_images"],
            "mean_clip": round(np.mean(stats["clip_scores"]), 4) if stats["clip_scores"] else 0,
            "std_clip": round(np.std(stats["clip_scores"]), 4) if stats["clip_scores"] else 0,
            "mean_attn": round(np.mean(stats["attn_scores"]), 4) if stats["attn_scores"] else 0,
            "std_attn": round(np.std(stats["attn_scores"]), 4) if stats["attn_scores"] else 0,
            "mean_combined": round(np.mean(stats["combined_scores"]), 4) if stats["combined_scores"] else 0,
            "mean_iou": round(np.mean(stats["iou_scores"]), 4) if stats["iou_scores"] else 0,
            "std_iou": round(np.std(stats["iou_scores"]), 4) if stats["iou_scores"] else 0,
        }

    csv_path = os.path.join(output_dir, "summary_table.csv")
    with open(csv_path, "w") as f:
        f.write("Category,Prompts,Images,CLIP (mean),CLIP (std),Attention (mean),Attention (std),Combined (mean),IoU (mean),IoU (std)\n")
        for cat, s in summary.items():
            f.write(f"{s['label']},{s['num_prompts']},{s['num_images']},{s['mean_clip']},{s['std_clip']},"
                    f"{s['mean_attn']},{s['std_attn']},{s['mean_combined']},{s['mean_iou']},{s['std_iou']}\n")
    print(f"Summary table saved -> {csv_path}")
    return summary


def plot_attn_vs_iou(results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        if not r["iou"] or "per_image" not in r["iou"]:
            continue
        cat = r["category"]
        color = CATEGORY_COLORS.get(cat, "#ffffff")
        for i, img_result in enumerate(r["scores"]):
            attn_score = img_result["scores"]["attention_score"]
            if i < len(r["iou"]["per_image"]):
                iou = r["iou"]["per_image"][i].get("mean_iou", 0)
                ax.scatter(attn_score, iou, c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

    for cat, color in CATEGORY_COLORS.items():
        ax.scatter([], [], c=color, label=CATEGORY_LABELS.get(cat, cat), s=40, edgecolors='white', linewidth=0.5)

    all_attn, all_iou = _collect_attn_iou_pairs(results)
    if len(all_attn) > 2:
        corr = np.corrcoef(all_attn, all_iou)[0, 1]
        z = np.polyfit(all_attn, all_iou, 1)
        x_line = np.linspace(min(all_attn), max(all_attn), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), '--', color='#ff9f43', alpha=0.8, linewidth=2, label=f'r = {corr:.3f}')

    ax.set_xlabel("Attention Alignment Score")
    ax.set_ylabel("IoU with CLIPSeg")
    ax.set_title("Attention Alignment vs Ground-Truth Segmentation (IoU)")
    ax.legend(loc='best', framealpha=0.8, facecolor='#1a1a2e')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_attn_vs_iou.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Attention vs IoU plot saved -> {path}")


def plot_clip_vs_combined(results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        color = CATEGORY_COLORS.get(r["category"], "#ffffff")
        for img_result in r["scores"]:
            ax.scatter(img_result["scores"]["clip_score"], img_result["scores"]["combined"],
                      c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    for cat, color in CATEGORY_COLORS.items():
        ax.scatter([], [], c=color, label=CATEGORY_LABELS.get(cat, cat), s=40, edgecolors='white', linewidth=0.5)
    ax.set_xlabel("CLIP Score (Global)")
    ax.set_ylabel("Combined Score")
    ax.set_title("CLIP Score vs Combined Score")
    ax.legend(loc='best', framealpha=0.8, facecolor='#1a1a2e')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_clip_vs_combined.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"CLIP vs Combined plot saved -> {path}")


def plot_iou_by_category(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    category_ious = defaultdict(list)
    for r in results:
        if not r["iou"] or "per_image" not in r["iou"]:
            continue
        for img_iou in r["iou"]["per_image"]:
            if "mean_iou" in img_iou:
                category_ious[r["category"]].append(img_iou["mean_iou"])

    categories = sorted(category_ious.keys())
    means = [np.mean(category_ious[c]) for c in categories]
    stds = [np.std(category_ious[c]) for c in categories]
    colors = [CATEGORY_COLORS.get(c, "#ffffff") for c in categories]
    labels = [CATEGORY_LABELS.get(c, c) for c in categories]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, color='#e0e0e0', fontweight='bold')
    ax.set_ylabel("Mean IoU with CLIPSeg")
    ax.set_title("Attention Map IoU by Prompt Category")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(means) * 1.3 if means else 1.0)
    plt.tight_layout()
    path = os.path.join(output_dir, "iou_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"IoU by category plot saved -> {path}")


def plot_token_iou_distribution(results, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    token_ious = defaultdict(list)
    for r in results:
        if not r["iou"] or "per_image" not in r["iou"]:
            continue
        for img_iou in r["iou"]["per_image"]:
            for tok, iou_val in img_iou.get("per_token", {}).items():
                token_ious[tok].append(iou_val)

    if not token_ious:
        plt.close(fig)
        return

    sorted_tokens = sorted(token_ious.keys(), key=lambda t: np.median(token_ious[t]), reverse=True)[:20]
    data = [token_ious[t] for t in sorted_tokens]

    bp = ax.boxplot(data, labels=sorted_tokens, patch_artist=True,
                    medianprops=dict(color='#ff9f43', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='#ff6b6b', markersize=4, alpha=0.5))
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_tokens)))
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("IoU with CLIPSeg")
    ax.set_title("Per-Token IoU Distribution (Top 20)")
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(output_dir, "token_iou_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Token IoU distribution plot saved -> {path}")


def plot_best_of_n_analysis(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    agreements = {"combined": 0, "clip_only": 0, "total": 0}
    combined_best_ious, clip_best_ious, random_ious = [], [], []

    for r in results:
        if not r["iou"] or "per_image" not in r["iou"]:
            continue
        n_images = min(len(r["scores"]), len(r["iou"]["per_image"]))
        if n_images < 2:
            continue
        ious = [r["iou"]["per_image"][i].get("mean_iou", 0) for i in range(n_images)]
        combined_scores = [r["scores"][i]["scores"]["combined"] for i in range(n_images)]
        clip_scores = [r["scores"][i]["scores"]["clip_score"] for i in range(n_images)]

        best_iou_idx = np.argmax(ious)
        agreements["total"] += 1
        if np.argmax(combined_scores) == best_iou_idx:
            agreements["combined"] += 1
        if np.argmax(clip_scores) == best_iou_idx:
            agreements["clip_only"] += 1

        combined_best_ious.append(ious[np.argmax(combined_scores)])
        clip_best_ious.append(ious[np.argmax(clip_scores)])
        random_ious.append(np.mean(ious))

    if agreements["total"] == 0:
        plt.close(fig)
        return

    methods = ["Combined\n(Ours)", "CLIP Only", "Random"]
    rates = [agreements["combined"]/agreements["total"], agreements["clip_only"]/agreements["total"], 1.0/8]
    colors = ["#4ecdc4", "#45b7d1", "#95a5a6"]

    bars = axes[0].bar(methods, rates, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=11, color='#e0e0e0', fontweight='bold')
    axes[0].set_ylabel("Agreement with Best IoU")
    axes[0].set_title("Best-of-N Selection: Agreement Rate")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, axis='y', alpha=0.3)

    mean_ious = [np.mean(combined_best_ious), np.mean(clip_best_ious), np.mean(random_ious)]
    bars2 = axes[1].bar(methods, mean_ious, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, m in zip(bars2, mean_ious):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=11, color='#e0e0e0', fontweight='bold')
    axes[1].set_ylabel("Mean IoU of Selected Image")
    axes[1].set_title("Best-of-N Selection: Quality")
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.suptitle("Best-of-N Re-Ranking Analysis", fontsize=14, y=1.02, color='#e0e0e0')
    plt.tight_layout()
    path = os.path.join(output_dir, "best_of_n_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Best-of-N analysis plot saved -> {path}")


def _collect_attn_iou_pairs(results):
    all_attn, all_iou = [], []
    for r in results:
        if not r["iou"] or "per_image" not in r["iou"]:
            continue
        n = min(len(r["scores"]), len(r["iou"]["per_image"]))
        for i in range(n):
            all_attn.append(r["scores"][i]["scores"]["attention_score"])
            all_iou.append(r["iou"]["per_image"][i].get("mean_iou", 0))
    return all_attn, all_iou


def run_analysis(results_dir: str):
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nRUNNING ANALYSIS\n{'=' * 60}")
    results = load_benchmark_results(results_dir)
    print(f"Loaded {len(results)} prompt results")
    if not results:
        print("No results found!")
        return

    summary = generate_summary_table(results, analysis_dir)
    plot_attn_vs_iou(results, analysis_dir)
    plot_clip_vs_combined(results, analysis_dir)
    plot_iou_by_category(results, analysis_dir)
    plot_token_iou_distribution(results, analysis_dir)
    plot_best_of_n_analysis(results, analysis_dir)

    report = {"num_prompts": len(results), "num_images": sum(len(r["scores"]) for r in results), "categories": summary}
    all_attn, all_iou = _collect_attn_iou_pairs(results)
    if len(all_attn) > 2:
        report["attn_iou_correlation"] = round(float(np.corrcoef(all_attn, all_iou)[0, 1]), 4)

    report_path = os.path.join(results_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report saved -> {report_path}")

    print(f"\n{'=' * 60}\nSUMMARY BY CATEGORY\n{'=' * 60}")
    print(f"{'Category':<16} {'CLIP':>8} {'Attn':>8} {'Combined':>10} {'IoU':>8}")
    print("-" * 54)
    for cat, s in summary.items():
        print(f"{s['label']:<16} {s['mean_clip']:>8.4f} {s['mean_attn']:>8.4f} {s['mean_combined']:>10.4f} {s['mean_iou']:>8.4f}")
    if "attn_iou_correlation" in report:
        print(f"\nAttention-IoU Pearson r = {report['attn_iou_correlation']:.4f}")
    print(f"\nAll outputs saved to: {analysis_dir}")
