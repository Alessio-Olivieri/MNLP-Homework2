"""
translation_finetune_opus.py
===========================

Fine‑tune a multilingual seq‑to‑seq model (e.g., OPUS‑MT) to translate
Italian *arcaico* → Italian *moderno* using a custom dataset.

Expected CSV / DataFrame columns
--------------------------------
* ``TESTO``                – source sentence in archaic Italian
* ``TRADUZIONE ITALIANA``  – modern Italian reference

Run
---
python translation_finetune_opus.py --csv data.csv --model Helsinki-NLP/opus-mt-it-it
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path, help="CSV or TSV file with the dataset")
    p.add_argument("--model", default="Helsinki-NLP/opus-mt-it-it", help="Base model")
    p.add_argument("--output_dir", default="finetuned_model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()

def load_dataset(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df = df.dropna(subset=["TESTO", "TRADUZIONE ITALIANA"])
    return Dataset.from_pandas(df)

def tokenize_function(examples, tokenizer):
    inputs = examples["TESTO"]
    targets = examples["TRADUZIONE ITALIANA"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    args = parse_args()
    raw_dataset = load_dataset(args.csv)
    split = raw_dataset.train_test_split(test_size=0.05, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = split.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=raw_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()