#!/usr/bin/env python3
"""
Push dataset and/or fine-tuned model to HuggingFace Hub.

Usage:
  # Push dataset only
  python scripts/push_to_hub.py --dataset --repo YOUR_USERNAME/vit-pipeline-dataset

  # Push model only
  python scripts/push_to_hub.py --model --repo YOUR_USERNAME/llava-vit-finetuned \
      --model-path outputs/lora_vit/merged

  # Push both
  python scripts/push_to_hub.py --dataset --model \
      --repo YOUR_USERNAME/vit-pipeline-dataset \
      --model-repo YOUR_USERNAME/llava-vit-finetuned \
      --model-path outputs/lora_vit/merged
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_logger, load_config, load_jsonl

log = get_logger("hub_push")


def push_dataset(records: list[dict], repo_id: str, private: bool = False) -> None:
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

    log.info(f"Preparing dataset for Hub upload ({len(records)} samples)...")

    # Build rows — include image bytes for a fully self-contained HF dataset
    rows = []
    for r in records:
        row = {
            "id":          r.get("id", ""),
            "instruction": r.get("instruction", ""),
            "response":    r.get("response", ""),
            "type":        r.get("type", ""),
            "source":      r.get("source", ""),
            "clip_score":  float(r.get("clip_score", 0.0)),
        }
        # Attach image path (HF datasets will encode as PIL Image if image column)
        img_path = r.get("image", "")
        if os.path.exists(img_path):
            row["image"] = img_path
        rows.append(row)

    ds = DatasetDict({
        "train": Dataset.from_list(rows).cast_column("image", HFImage())
    })

    log.info(f"Pushing to Hub: {repo_id} (private={private})...")
    ds.push_to_hub(repo_id, private=private)
    log.info(f"Dataset live at: https://huggingface.co/datasets/{repo_id}")


def push_model(model_path: str, repo_id: str, private: bool = False) -> None:
    from transformers import AutoModelForCausalLM, AutoProcessor

    log.info(f"Loading merged model from {model_path}...")
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    log.info(f"Pushing model to Hub: {repo_id}...")
    model.push_to_hub(repo_id, private=private)
    processor.push_to_hub(repo_id, private=private)
    log.info(f"Model live at: https://huggingface.co/models/{repo_id}")


def write_model_card(repo_id: str, dataset_repo: str, stats: dict) -> None:
    """Write a model card for the fine-tuned VLM."""
    card = f"""---
license: apache-2.0
base_model: llava-hf/llava-v1.6-mistral-7b-hf
datasets:
  - {dataset_repo}
tags:
  - vision-language-model
  - instruction-tuning
  - lora
  - llava
---

# LLaVA-1.6 Fine-tuned on Automated VIT Pipeline Data

This model is fine-tuned from [LLaVA-1.6-Mistral-7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) using a fully automated visual instruction tuning data pipeline (no GPT-4 annotation required).

## Training Data

The training dataset was built using an open-source pipeline:
- **Images**: COCO 2017 + Open Images VSR + SDXL synthetic images
- **Annotation**: Qwen2-VL-7B-Instruct (automatic dense annotation)
- **Instruction generation**: Llama 3.1 8B via Ollama (local, no API cost)
- **Quality filtering**: CLIP score + NER-based hallucination check + cosine dedup

**Dataset stats:**
- Total samples: {stats.get('total', 'N/A')}
- Instruction types: {', '.join(stats.get('by_type', {}).keys())}
- Avg CLIP score: {stats.get('clip_score', {}).get('mean', 'N/A')}

## Usage

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

model = LlavaNextForConditionalGeneration.from_pretrained(
    "{repo_id}", torch_dtype=torch.float16, device_map="auto"
)
processor = LlavaNextProcessor.from_pretrained("{repo_id}")

image = Image.open("your_image.jpg")
prompt = "[INST] <image>\\nDescribe what is happening in this image. [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Citation

This model was built as a research project demonstrating automated visual instruction tuning data generation.
"""
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    log.info("Model card uploaded.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/pipeline_config.yaml")
    p.add_argument("--dataset",     action="store_true", help="Push dataset")
    p.add_argument("--model",       action="store_true", help="Push model")
    p.add_argument("--repo",        default=None,        help="Dataset repo ID")
    p.add_argument("--model-repo",  default=None,        help="Model repo ID")
    p.add_argument("--model-path",  default="outputs/lora_vit/merged")
    p.add_argument("--private",     action="store_true", help="Make repos private")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if not args.dataset and not args.model:
        print("Specify at least one of --dataset or --model")
        sys.exit(1)

    # Check HF token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set. Run 'huggingface-cli login' first.")

    # Load dataset stats
    stats = {}
    stats_path = "results/dataset_stats.json"
    if os.path.exists(stats_path):
        import json
        with open(stats_path) as f:
            stats = json.load(f)

    if args.dataset:
        repo = args.repo or cfg.get("hub", {}).get("repo_id")
        if not repo or "YOUR_USERNAME" in repo:
            log.error("Provide a valid --repo or set hub.repo_id in config")
            sys.exit(1)
        filtered_path = os.path.join(cfg["paths"]["filtered"], "filtered.jsonl")
        records = load_jsonl(filtered_path)
        push_dataset(records, repo, private=args.private)

    if args.model:
        repo = args.model_repo or cfg.get("hub", {}).get("model_repo_id")
        if not repo or "YOUR_USERNAME" in repo:
            log.error("Provide a valid --model-repo")
            sys.exit(1)
        push_model(args.model_path, repo, private=args.private)
        dataset_repo = args.repo or cfg.get("hub", {}).get("repo_id", "")
        write_model_card(repo, dataset_repo, stats)

    log.info("Hub upload complete.")


if __name__ == "__main__":
    main()
