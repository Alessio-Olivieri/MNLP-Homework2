"""
minerva3b_loader.py
===================

Loader for *Minerva‑3B‑base* model.

Example
-------
>>> from minerva3b_loader import load_pipeline
>>> pipe = load_pipeline()
>>> pipe("Spiega il Teorema di Pitagora.", max_new_tokens=64)
"""

import torch
from transformers import pipeline

MODEL_ID = "sapienzanlp/Minerva-3B-base-v1.0"

def load_pipeline(device_map: str | dict = "auto"):
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device_map,
    )