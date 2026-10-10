import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    )
    model.save_pretrained(ROOT / ".cache" / "model")
    tokenizer.save_pretrained(ROOT / ".cache" / "model")
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"])
    torch.save(tokens, ROOT / ".cache" / "tokens.pt")
    manifest = {
        "model": MODEL,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "token_count": tokens.numel(),
    }
    (ROOT / ".cache" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
