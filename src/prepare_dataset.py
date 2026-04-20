"""
Stage 5 — Dataset Preparation
Converts filtered.jsonl → LLaMA-Factory format → fine-tuning ready.
Optionally pushes to HuggingFace Hub.
"""
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from .utils import get_logger, load_json, load_jsonl, save_json

log = get_logger(__name__)


def to_llama_factory_format(records: list[dict]) -> list[dict]:
    """
    Convert to LLaMA-Factory sharegpt format for multimodal VLM fine-tuning.
    See: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
    """
    out = []
    for rec in records:
        out.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\n{rec['instruction']}"
                },
                {
                    "role": "assistant",
                    "content": rec["response"]
                }
            ],
            "images": [os.path.abspath(rec["image"])],
            "_meta": {
                "id":         rec.get("id", ""),
                "type":       rec.get("type", ""),
                "source":     rec.get("source", ""),
                "clip_score": rec.get("clip_score", 0.0),
            }
        })
    return out


def write_dataset_info(dataset_name: str, data_path: str, info_path: str) -> None:
    """Append our dataset entry to LLaMA-Factory's dataset_info.json."""
    # Load existing info or start fresh
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    info[dataset_name] = {
        "file_name": os.path.basename(data_path),
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images":   "images"
        }
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    log.info(f"[Prepare] dataset_info.json updated at {info_path}")


def create_llama_factory_yaml(cfg: dict) -> str:
    """Write the LoRA training config YAML for LLaMA-Factory."""
    dataset_abs = os.path.abspath(cfg["paths"]["llama_factory_dataset"])
    out_dir_abs = os.path.abspath(cfg["training"]["output_dir"])

    yaml_content = f"""### Model
model_name_or_path: {cfg['training']['base_model']}
trust_remote_code: true

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {cfg['training']['lora_rank']}
lora_alpha: {cfg['training']['lora_alpha']}
lora_target: {cfg['training']['lora_target']}

### Dataset
dataset: vit_dataset
template: llava1_5
cutoff_len: {cfg['training']['cutoff_len']}
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 4

### Output
output_dir: {out_dir_abs}
logging_steps: 50
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Train
per_device_train_batch_size: {cfg['training']['batch_size']}
gradient_accumulation_steps: {cfg['training']['grad_accum']}
learning_rate: {cfg['training']['lr']}
num_train_epochs: {cfg['training']['num_epochs']}
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: {str(cfg['training']['bf16']).lower()}
ddp_timeout: 180000000

### Eval (optional — remove if no val split)
# val_size: 0.01
# evaluation_strategy: steps
# eval_steps: 500
# per_device_eval_batch_size: 1
# load_best_model_at_end: true
"""
    yaml_path = "configs/train_lora.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    log.info(f"[Prepare] Training YAML written to {yaml_path}")
    return yaml_path


def push_to_hub(records: list[dict], cfg: dict) -> None:
    """Push the dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    repo_id = cfg["hub"]["repo_id"]
    log.info(f"[Hub] Pushing dataset to {repo_id}...")

    # Build a flat dataset (no image bytes — just paths + metadata)
    flat = [{
        "id":          r.get("id", ""),
        "image_path":  r.get("image", ""),
        "instruction": r.get("instruction", ""),
        "response":    r.get("response", ""),
        "type":        r.get("type", ""),
        "source":      r.get("source", ""),
        "clip_score":  r.get("clip_score", 0.0),
    } for r in records]

    ds = DatasetDict({"train": Dataset.from_list(flat)})
    ds.push_to_hub(repo_id, private=False)
    log.info(f"[Hub] Dataset pushed: https://huggingface.co/datasets/{repo_id}")


def run(cfg: dict) -> None:
    in_path    = os.path.join(cfg["paths"]["filtered"], "filtered.jsonl")
    final_path = cfg["paths"]["final_dataset"]
    lf_path    = cfg["paths"]["llama_factory_dataset"]
    info_path  = cfg["paths"]["llama_factory_data_info"]

    records = load_jsonl(in_path)
    log.info(f"[Stage 5] Preparing {len(records)} samples for fine-tuning...")

    # Convert to LLaMA-Factory format
    lf_records = to_llama_factory_format(records)

    # Save final dataset
    save_json(lf_records, final_path)
    log.info(f"  Final dataset saved → {final_path}")

    # Copy to LLaMA-Factory data directory
    Path(lf_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(lf_records, lf_path)
    write_dataset_info("vit_dataset", lf_path, info_path)

    # Write training YAML
    yaml_path = create_llama_factory_yaml(cfg)

    # Print instructions
    log.info("\n" + "="*60)
    log.info("  FINE-TUNING INSTRUCTIONS")
    log.info("="*60)
    log.info(f"  Dataset: {len(lf_records)} samples ready")
    log.info(f"  Run fine-tuning with:")
    log.info(f"    cd LLaMA-Factory")
    log.info(f"    llamafactory-cli train ../{yaml_path}")
    log.info("="*60 + "\n")

    # Optional HF Hub upload
    if cfg.get("hub", {}).get("push_dataset", False):
        push_to_hub(records, cfg)

    log.info("[Stage 5 done]")
