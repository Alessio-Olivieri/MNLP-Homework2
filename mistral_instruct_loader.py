"""
mistral_instruct_loader.py
==========================

Utility to load *Mixtral‑8x7B‑Instruct* (FP16) for generation.

Example
-------
>>> from mistral_instruct_loader import load_model, chat
>>> model, tokenizer = load_model()
>>> response = chat(model, tokenizer, "What is your favourite condiment?")
"""

from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def load_model(device_map: str | dict = "auto") -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load the Mixtral-8×7B-Instruct model in half‑precision on the specified devices."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer

def chat(model, tokenizer, prompt: str, history: List[dict] | None = None, max_new_tokens: int = 128) -> str:
    """Simple chat wrapper around *generate* using the HF chat template."""
    if history is None:
        history = []
    messages = history + [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)