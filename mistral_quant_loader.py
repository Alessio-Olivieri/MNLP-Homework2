"""
mistral_quant_loader.py
=======================

Load Mixtral‑8x7B‑Instruct with 4‑bit quantization via *bitsandbytes*.

Requires:
    pip install --no-deps xformers trl peft accelerate bitsandbytes

Example
-------
>>> from mistral_quant_loader import load_model, chat
>>> model, tokenizer = load_model()
>>> chat(model, tokenizer, "Hello!")
"""

from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def _bnb_config() -> BitsAndBytesConfig:
    """Return a 4‑bit *bitsandbytes* configuration tuned for current GPUs."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_model(device_map: str | dict = "auto") -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load the quantised model using the config above."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=_bnb_config(),
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer

def chat(model, tokenizer, prompt: str, history: List[dict] | None = None, max_new_tokens: int = 128) -> str:
    from mistral_instruct_loader import chat as _chat  # reuse implementation
    return _chat(model, tokenizer, prompt, history, max_new_tokens)