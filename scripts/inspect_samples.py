#!/usr/bin/env python3
"""
Interactive sample inspector.
Browse generated instruction pairs, see the image + annotation + QA side-by-side.
Useful for spotting systematic quality issues before fine-tuning.

Usage:
  python scripts/inspect_samples.py --stage filtered   # review filtered pairs
  python scripts/inspect_samples.py --stage raw        # review raw instructions
  python scripts/inspect_samples.py --type reasoning   # filter by type
  python scripts/inspect_samples.py --n 50             # sample 50 at random
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.utils import load_config, load_jsonl

console = Console()


def show_sample(rec: dict, idx: int, total: int) -> None:
    """Pretty-print one sample in the terminal."""
    console.rule(f"[bold]Sample {idx+1}/{total}[/bold]")

    # Metadata row
    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    meta.add_column("key",   style="dim", width=14)
    meta.add_column("value", style="white")
    meta.add_row("ID",         rec.get("id", "-"))
    meta.add_row("Type",       f"[cyan]{rec.get('type', '-')}[/cyan]")
    meta.add_row("Source",     rec.get("source", "-"))
    meta.add_row("CLIP score", f"{rec.get('clip_score', '-')}")
    meta.add_row("Halu score", f"{rec.get('h_score', '-')}")
    meta.add_row("Image",      rec.get("image", "-"))
    console.print(meta)

    # Instruction / Response
    console.print(Panel(rec["instruction"], title="[bold green]Instruction[/bold green]", border_style="green"))
    console.print(Panel(rec["response"],    title="[bold blue]Response[/bold blue]",      border_style="blue"))

    # Annotation snippet
    ann = rec.get("annotation", {})
    if ann:
        ann_lines = []
        if ann.get("dense_caption"):
            ann_lines.append(f"Caption: {ann['dense_caption'][:200]}")
        if ann.get("objects"):
            objs = [o.get("name", "?") if isinstance(o, dict) else str(o) for o in ann["objects"][:5]]
            ann_lines.append(f"Objects: {', '.join(objs)}")
        if ann.get("scene"):
            ann_lines.append(f"Scene: {ann['scene']}")
        if ann.get("text_in_image"):
            ann_lines.append(f"Text in image: {ann['text_in_image'][:3]}")
        if ann_lines:
            console.print(Panel("\n".join(ann_lines), title="[dim]Annotation[/dim]", border_style="dim"))

    # Try to open the image
    img_path = rec.get("image", "")
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            console.print(f"[dim]Image: {img.width}×{img.height} px[/dim]")
            # Show image in terminal if iTerm2 / kitty protocol available
            # (Falls back silently if not supported)
            try:
                import subprocess
                subprocess.run(["imgcat", img_path], capture_output=True, timeout=2)
            except Exception:
                pass
        except Exception:
            pass
    else:
        console.print("[dim red]Image file not found on disk.[/dim red]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/pipeline_config.yaml")
    p.add_argument("--stage",   default="filtered", choices=["filtered", "instructions"],
                   help="Which JSONL file to inspect")
    p.add_argument("--type",    default=None,
                   help="Filter to a specific instruction type")
    p.add_argument("--source",  default=None,
                   help="Filter to a specific source (coco, open_images_vsr, sdxl_synthetic)")
    p.add_argument("--n",       type=int, default=None,
                   help="Number of samples to inspect (random). Default: all")
    p.add_argument("--min-clip", type=float, default=None,
                   help="Show only samples with CLIP score ≥ this value")
    p.add_argument("--max-clip", type=float, default=None,
                   help="Show only samples with CLIP score ≤ this value")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Resolve path
    if args.stage == "filtered":
        path = os.path.join(cfg["paths"]["filtered"], "filtered.jsonl")
    else:
        path = os.path.join(cfg["paths"]["instructions"], "instructions.jsonl")

    if not os.path.exists(path):
        console.print(f"[red]File not found: {path}[/red]")
        console.print("Run earlier pipeline stages first.")
        sys.exit(1)

    records = load_jsonl(path)
    console.print(f"[green]Loaded {len(records)} records from {path}[/green]")

    # Apply filters
    if args.type:
        records = [r for r in records if r.get("type") == args.type]
        console.print(f"  → {len(records)} after type='{args.type}' filter")
    if args.source:
        records = [r for r in records if r.get("source") == args.source]
        console.print(f"  → {len(records)} after source='{args.source}' filter")
    if args.min_clip is not None:
        records = [r for r in records if r.get("clip_score", 0) >= args.min_clip]
        console.print(f"  → {len(records)} after min_clip={args.min_clip} filter")
    if args.max_clip is not None:
        records = [r for r in records if r.get("clip_score", 1) <= args.max_clip]
        console.print(f"  → {len(records)} after max_clip={args.max_clip} filter")

    if not records:
        console.print("[red]No records match your filters.[/red]")
        sys.exit(0)

    # Sampling
    if args.n and args.n < len(records):
        records = random.sample(records, args.n)
        console.print(f"  → Randomly sampled {len(records)}")

    # Summary stats
    from collections import Counter
    import numpy as np
    type_counts = Counter(r.get("type") for r in records)
    clips = [r.get("clip_score", 0) for r in records if r.get("clip_score")]
    console.print(f"\nType distribution: {dict(type_counts)}")
    if clips:
        console.print(f"CLIP score: mean={np.mean(clips):.3f}  min={np.min(clips):.3f}  max={np.max(clips):.3f}\n")

    # Interactive browse
    i = 0
    while i < len(records):
        show_sample(records[i], i, len(records))
        console.print("\n[dim]Enter: next | p: previous | s: skip 10 | q: quit[/dim]")
        try:
            key = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "q":
            break
        elif key == "p":
            i = max(0, i - 1)
        elif key == "s":
            i = min(len(records) - 1, i + 10)
        else:
            i += 1

    console.print("[green]Inspector closed.[/green]")


if __name__ == "__main__":
    main()
