#!/usr/bin/env python3
"""
Smoke test — verify all pipeline components before running the full pipeline.
Tests: GPU, Qwen2-VL, Ollama/Llama, CLIP, spaCy, HuggingFace datasets.
Run before starting: python scripts/quick_test.py

Each test is independent — failing one won't stop the others.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

results = []


def test(name: str):
    """Decorator to wrap each test."""
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                info = fn()
                elapsed = time.time() - t0
                results.append((name, "PASS", info or "", f"{elapsed:.1f}s"))
            except Exception as e:
                elapsed = time.time() - t0
                results.append((name, "FAIL", str(e)[:80], f"{elapsed:.1f}s"))
        return wrapper
    return decorator


@test("GPU")
def check_gpu():
    import torch
    assert torch.cuda.is_available(), "No CUDA GPU found"
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    return f"{name} ({mem}GB VRAM)"


@test("PyTorch")
def check_torch():
    import torch
    return f"PyTorch {torch.__version__}"


@test("Transformers")
def check_transformers():
    import transformers
    return f"transformers {transformers.__version__}"


@test("BitsAndBytes (4-bit)")
def check_bnb():
    import bitsandbytes
    import torch
    # Quick 4-bit layer test
    layer = bitsandbytes.nn.Linear4bit(16, 16)
    return f"bitsandbytes {bitsandbytes.__version__}"


@test("open_clip (CLIP filter)")
def check_clip():
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tok = tokenizer(["a test image"])
    with torch.no_grad():
        feat = model.encode_text(tok)
    return f"open_clip, text feat shape {feat.shape}"


@test("spaCy NER")
def check_spacy():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("A cat sat on the mat next to a red bowl.")
    nouns = [t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN")]
    return f"spaCy en_core_web_sm — nouns: {nouns}"


@test("HuggingFace datasets (COCO preview)")
def check_datasets():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/COCO", split="train[:2]", trust_remote_code=True)
    return f"{len(ds)} sample(s) loaded"


@test("Diffusers (SDXL available)")
def check_diffusers():
    import diffusers
    return f"diffusers {diffusers.__version__}"


@test("Ollama (Llama 3.1)")
def check_ollama():
    import requests
    resp = requests.get("http://localhost:11434", timeout=5)
    # Check model is available
    models = requests.get("http://localhost:11434/api/tags", timeout=5).json()
    names = [m["name"] for m in models.get("models", [])]
    llama = [n for n in names if "llama3.1" in n or "llama3" in n]
    assert llama, f"llama3.1:8b not found. Available: {names}"
    # Quick generation test
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": 'Return only: {"test": "ok"}',
        "stream": False,
        "options": {"num_predict": 20},
    }, timeout=30)
    return f"Ollama models: {llama}"


@test("qwen-vl-utils")
def check_qwen_vl_utils():
    import qwen_vl_utils
    return "qwen-vl-utils available"


@test("Qwen2-VL model download (metadata only)")
def check_qwen2vl():
    from huggingface_hub import model_info
    info = model_info("Qwen/Qwen2-VL-7B-Instruct")
    return f"Model ID: {info.id}, tags: {info.tags[:3]}"


@test("rich + yaml + tqdm")
def check_utils():
    import yaml, tqdm, rich
    return f"rich {rich.__version__}"


def main():
    console.print("\n[bold]VIT Pipeline — Smoke Tests[/bold]\n")

    # Run all tests
    for fn in [
        check_gpu, check_torch, check_transformers, check_bnb,
        check_clip, check_spacy, check_datasets, check_diffusers,
        check_ollama, check_qwen_vl_utils, check_qwen2vl, check_utils,
    ]:
        fn()

    # Print summary table
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Component",   style="white",  width=30)
    table.add_column("Status",      style="bold",   width=8)
    table.add_column("Info",        style="dim",    width=50)
    table.add_column("Time",        style="dim",    width=7)

    passed = 0
    for name, status, info, t in results:
        colour = "green" if status == "PASS" else "red"
        table.add_row(name, f"[{colour}]{status}[/{colour}]", info, t)
        if status == "PASS":
            passed += 1

    console.print(table)
    console.print(f"\n[bold]{passed}/{len(results)} tests passed.[/bold]")

    if passed < len(results):
        console.print("[yellow]Fix failing tests before running the pipeline.[/yellow]")
        console.print("Most common fixes:")
        console.print("  Ollama not running → ollama serve &")
        console.print("  Llama not pulled   → ollama pull llama3.1:8b")
        console.print("  spaCy missing      → python -m spacy download en_core_web_sm")
        sys.exit(1)
    else:
        console.print("[green]All tests passed. Ready to run the pipeline![/green]")
        console.print("  python scripts/run_pipeline.py --config configs/pipeline_config.yaml")


if __name__ == "__main__":
    main()
