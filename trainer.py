import json
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from transformers import EarlyStoppingCallback


# 1. Load and map labels
with open("dataset/generated_train.json", "r") as f:
    raw_data = json.load(f)

label_map = {"noCode": 0, "containsCode": 1}
for entry in raw_data:
    entry["label"] = label_map[entry["label"]]

# 2. Convert to HuggingFace Dataset
dataset = Dataset.from_list(raw_data)
dataset = dataset.train_test_split(test_size=0.2)

# 3. Load CodeBERT tokenizer
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# 4. Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(tokenize)

# 5. Load model
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Define training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 8. Train!
trainer.train()

# 9. Save model
trainer.save_model("code-detector-model")
