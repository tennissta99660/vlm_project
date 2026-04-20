"""
Shared utilities for the VIT pipeline.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

console = Console()


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
    )
    return logging.getLogger(name)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(record: dict, path: str) -> None:
    """Append one record to a .jsonl file (for streaming saves)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> set:
    """Return a set of already-processed IDs from a checkpoint file."""
    if not os.path.exists(ckpt_path):
        return set()
    done = set()
    with open(ckpt_path) as f:
        for line in f:
            done.add(line.strip())
    return done


def save_checkpoint(item_id: str, ckpt_path: str) -> None:
    with open(ckpt_path, "a") as f:
        f.write(item_id + "\n")


# ── Retry decorator ───────────────────────────────────────────────────────────

def retry(times: int = 3, delay: float = 2.0):
    """Decorator: retry a function up to `times` on any Exception."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator


# ── Ensure dirs ───────────────────────────────────────────────────────────────

def ensure_dirs(cfg: dict) -> None:
    for key, path in cfg["paths"].items():
        if not path.endswith(".json"):
            Path(path).mkdir(parents=True, exist_ok=True)
