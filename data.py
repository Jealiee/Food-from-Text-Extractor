from transformers import AutoTokenizer
from datasets import Dataset
import json


def tokenize_dataset(model_name, json_paths):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if isinstance(json_paths, str):
        json_paths = [json_paths]

    data = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))

    ds = Dataset.from_list(data)

    # 90/10 split
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    def preprocess(example):
        input_text = "extract_food: " + example["input"]
        inputs = tokenizer(
            input_text, truncation=True, padding="max_length", max_length=256
        )
        labels = tokenizer(
            example["output"], truncation=True, padding="max_length", max_length=128
        )

        labels_ids = [
            token if token != tokenizer.pad_token_id else -100
            for token in labels["input_ids"]
        ]

        inputs["labels"] = labels_ids
        return inputs

    
    train_ds = train_ds.map(preprocess, batched=False)
    val_ds = val_ds.map(preprocess, batched=False)

    return train_ds, val_ds
