from transformers import AutoTokenizer
from datasets import Dataset
import json


def tokenize_dataset(model_name, json_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # generated with LLM to mimic random user inputs
    with open(json_path, "r") as f:
        data = json.load(f)

    ds = Dataset.from_list(data)

    # 90/10 split
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    def preprocess(example):
        input_text = "extract food:" + example["input"]
        inputs = tokenizer(
            input_text, truncation=True, padding="max_length", max_length=128
        )
        labels = tokenizer(
            example["output"], truncation=True, padding="max_length", max_length=128
        )

        labels_ids = [
            token if token is not tokenizer.pad_token_id else -100
            for token in labels["input_ids"]
        ]

        inputs["labels"] = labels_ids
        return inputs

    
    train_ds = train_ds.map(preprocess, batched=False)
    val_ds = val_ds.map(preprocess, batched=False)

    return train_ds, val_ds
