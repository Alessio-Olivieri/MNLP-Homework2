"""
minerva350_loader.py
====================

Light‑weight loader for *Minerva‑350M‑base* (FP16).

Example
-------
>>> from minerva350_loader import load_pipeline
>>> pipe = load_pipeline()
>>> pipe("Chi fu Cesare Augusto?", max_new_tokens=64)
"""

import torch
from transformers import pipeline

MODEL_ID = "sapienzanlp/Minerva-350M-base-v1.0"

def load_pipeline(device_map: str | dict = "auto"):
    """Return a generation pipeline with bfloat16 weights."""
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device_map,
    )