#!/usr/bin/env bash
# =============================================================================
#  VIT Pipeline — one-shot setup
#  Run once: bash setup.sh
#  Tested on: Kaggle (T4/P100), Colab (T4/A100), Ubuntu 22.04
# =============================================================================
set -euo pipefail

echo "════════════════════════════════════════════"
echo "  Visual Instruction Tuning Pipeline Setup  "
echo "════════════════════════════════════════════"

# ── 1. Python deps ────────────────────────────────────────────────────────────
echo "[1/6] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── 2. spaCy English model (for NER in hallucination filter) ─────────────────
echo "[2/6] Downloading spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm -q

# ── 3. Ollama (local Llama 3.1 8B — no API key needed) ───────────────────────
echo "[3/6] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[3/6] Starting Ollama daemon in background..."
ollama serve &> /tmp/ollama.log &
sleep 5   # wait for daemon to be ready

echo "[3/6] Pulling Llama 3.1 8B (this may take a few minutes)..."
ollama pull llama3.1:8b

# ── 4. LLaMA-Factory (fine-tuning framework) ─────────────────────────────────
echo "[4/6] Cloning LLaMA-Factory..."
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -q -e ".[torch,metrics]"
    cd ..
fi

# ── 5. lmms-eval (benchmarking) ───────────────────────────────────────────────
echo "[5/6] Cloning lmms-eval..."
if [ ! -d "lmms-eval" ]; then
    git clone --depth=1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    pip install -q -e .
    cd ..
fi

# ── 6. HuggingFace login (optional — needed to push dataset) ─────────────────
echo "[6/6] (Optional) HuggingFace Hub login..."
echo "  Run 'huggingface-cli login' manually to push your dataset to the Hub."

# ── Verify GPU ────────────────────────────────────────────────────────────────
python - <<'EOF'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU FOUND"
mem = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
print(f"\n  GPU : {gpu}")
print(f"  VRAM: {mem} GB")
if mem < 12:
    print("  WARNING: <12GB VRAM detected. Use 4-bit quantisation (set USE_4BIT=true in config).")
EOF

echo ""
echo "  Setup complete. Run the full pipeline with:"
echo "    python scripts/run_pipeline.py --config configs/pipeline_config.yaml"
echo ""
