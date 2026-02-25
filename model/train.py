from transformers import AutoModelForSeq2SeqLM
from model.data import tokenize_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback
import torch

model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

train_ds, val_ds = tokenize_dataset(model_name, ["food_ds.json", "food_ds_fuzzed.json"])

tokenizer = AutoTokenizer.from_pretrained("t5-small")
_sample = val_ds[0]

class PrintSampleCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 250 != 0 or state.global_step == 0:
            return
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                "extract_food: " + _sample["input"],
                return_tensors="pt"
            ).to(model.device)
            outputs = model.generate(**inputs, max_length=128)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[step {state.global_step}]")
        print(f"  input:     {_sample['input']}")
        print(f"  target:    {_sample['output']}")
        print(f"  generated: {generated}")
        model.train()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    learning_rate=5e-5,
    logging_steps=20,
    save_strategy="epoch",
    weight_decay=0.01,
    warmup_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    callbacks=[PrintSampleCallback()],
)
trainer.train()

for example in val_ds.select(range(5)):
    input_text = "extract_food: " + example["input"]
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Input:", example["input"])
    print("Target:", example["output"])
    print("Generated:", generated_text)
    print("---")
