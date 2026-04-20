#!/usr/bin/env bash
# Merge LoRA adapter into base model weights.
# Run this after training, before lmms-eval.
set -euo pipefail

BASE="llava-hf/llava-v1.6-mistral-7b-hf"
ADAPTER="outputs/lora_vit"
MERGED="outputs/lora_vit/merged"

echo "Merging LoRA adapter → $MERGED"

python - <<EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch, os

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "$BASE", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "$ADAPTER")
print("Merging weights...")
model = model.merge_and_unload()
print("Saving merged model...")
os.makedirs("$MERGED", exist_ok=True)
model.save_pretrained("$MERGED")
processor = AutoProcessor.from_pretrained("$BASE", trust_remote_code=True)
processor.save_pretrained("$MERGED")
print("Done → $MERGED")
EOF
