# Quickstart Cheat Sheet

## 0. First time only

```bash
git clone https://github.com/YOUR_USERNAME/vit_pipeline.git
cd vit_pipeline
bash setup.sh                          # installs everything (~10 min)
python scripts/quick_test.py           # verify all components work
```

## 1. Run the pipeline

```bash
# Full pipeline (all 6 stages)
python scripts/run_pipeline.py

# Stages separately (pipeline is resumable — safe to stop/restart)
python scripts/run_pipeline.py --stages 1          # collect images
python scripts/run_pipeline.py --stages 2          # annotate with Qwen2-VL
python scripts/run_pipeline.py --stages 3          # generate instructions
python scripts/run_pipeline.py --stages 4          # filter for quality
python scripts/run_pipeline.py --stages 5          # prepare for fine-tuning
```

## 2. Fine-tune

```bash
cd LLaMA-Factory
llamafactory-cli train ../configs/train_lora.yaml  # ~8-12h on T4
cd ..
bash scripts/merge_lora.sh                         # merge LoRA weights
```

## 3. Evaluate

```bash
# Baseline (no fine-tuning)
bash scripts/run_eval.sh llava-hf/llava-v1.6-mistral-7b-hf

# Your model
bash scripts/run_eval.sh outputs/lora_vit/merged
```

## 4. Inspect samples

```bash
python scripts/inspect_samples.py                  # browse all filtered samples
python scripts/inspect_samples.py --type reasoning # only reasoning type
python scripts/inspect_samples.py --n 20           # random 20 samples
```

## 5. Push to HuggingFace Hub

```bash
huggingface-cli login
python scripts/push_to_hub.py \
    --dataset --model \
    --repo YOUR_USERNAME/vit-pipeline-dataset \
    --model-repo YOUR_USERNAME/llava-vit-finetuned \
    --model-path outputs/lora_vit/merged
```

---

## Key files to edit

| File | What to change |
|------|---------------|
| `configs/pipeline_config.yaml` | Dataset size, VRAM settings, filtering thresholds |
| `src/generate_instructions.py` | `TYPE_PROMPTS` dict — add/edit instruction types |
| `src/annotate.py` | `ANNOTATION_PROMPT` — change what fields are extracted |
| `src/collect_data.py` | `SYNTHETIC_PROMPT_TEMPLATES` — add new image categories |

## VRAM guide

| GPU | VRAM | Setting |
|-----|------|---------|
| T4 (Kaggle/Colab free) | 15GB | `use_4bit: true` (default) |
| A10G (Colab Pro) | 24GB | `use_4bit: false` |
| A100 (Colab Pro+) | 40/80GB | `use_4bit: false`, larger batch |

## Troubleshooting

**Ollama not responding:**
```bash
ollama serve &   # start in background
sleep 3 && ollama list   # verify
```

**CUDA out of memory (Stage 2):**
```bash
# In config: use_4bit: true, batch_size: 1
# Also try: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Stage resumes from wrong point:**
```bash
# Delete the checkpoint file to restart a stage from scratch:
rm data/annotations/raw_meta.jsonl.ckpt   # Stage 1
rm data/annotations/annotated_meta.jsonl.ckpt  # Stage 2
```

**Llama JSON parse errors (Stage 3):**  
Normal for ~5-10% of samples — the pipeline skips bad parses and continues.

**LLaMA-Factory not finding dataset:**
```bash
# Make sure dataset_info.json was updated by Stage 5:
cat LLaMA-Factory/data/dataset_info.json | grep vit_dataset
```
