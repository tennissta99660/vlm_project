#!/usr/bin/env python3
"""
Master pipeline runner.
Usage:
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --stages 1 2
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --stages 4 5 6
"""
import argparse
import sys
import time
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dirs, get_logger, load_config

log = get_logger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Visual Instruction Tuning Pipeline")
    p.add_argument("--config",  default="configs/pipeline_config.yaml",
                   help="Path to pipeline_config.yaml")
    p.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                   help="Which stages to run (1-6). Default: all.")
    p.add_argument("--skip-synthetic", action="store_true",
                   help="Skip SDXL synthetic image generation (Stage 1)")
    return p.parse_args()


STAGE_NAMES = {
    1: "Data Collection",
    2: "Auto-Annotation (Qwen2-VL)",
    3: "Instruction Generation (Llama 3.1)",
    4: "Quality Filtering",
    5: "Dataset Preparation",
    6: "Evaluation & Ablation",
}


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    ensure_dirs(cfg)

    stages = sorted(args.stages)
    log.info(f"Running stages: {stages}")

    total_start = time.time()

    for stage_num in stages:
        name = STAGE_NAMES.get(stage_num, f"Stage {stage_num}")
        log.info(f"\n{'='*60}")
        log.info(f"  STAGE {stage_num}: {name}")
        log.info(f"{'='*60}")
        t0 = time.time()

        try:
            if stage_num == 1:
                from src import collect_data
                if args.skip_synthetic:
                    cfg["data"]["num_synthetic_images"] = 0
                collect_data.run(cfg)

            elif stage_num == 2:
                from src import annotate
                annotate.run(cfg)

            elif stage_num == 3:
                from src import generate_instructions
                generate_instructions.run(cfg)

            elif stage_num == 4:
                from src import quality_filter
                quality_filter.run(cfg)

            elif stage_num == 5:
                from src import prepare_dataset
                prepare_dataset.run(cfg)

            elif stage_num == 6:
                from src import evaluate
                evaluate.run(cfg)

            else:
                log.warning(f"Unknown stage {stage_num} — skipping.")
                continue

        except KeyboardInterrupt:
            log.warning(f"\nInterrupted at Stage {stage_num}. Progress is saved — re-run to resume.")
            sys.exit(0)
        except Exception as e:
            log.exception(f"Stage {stage_num} failed: {e}")
            log.info("Fix the error and re-run — completed stages will be skipped automatically.")
            sys.exit(1)

        elapsed = time.time() - t0
        log.info(f"  Stage {stage_num} complete in {elapsed/60:.1f} min")

    total = time.time() - total_start
    log.info(f"\nAll stages complete in {total/60:.1f} min total.")
    log.info("Next: cd LLaMA-Factory && llamafactory-cli train ../configs/train_lora.yaml")


if __name__ == "__main__":
    main()
