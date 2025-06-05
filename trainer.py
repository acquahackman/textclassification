import json
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from transformers import EarlyStoppingCallback
import os

# Create output directories in the home directory where you should have write permissions
home_dir = os.path.expanduser("~")
output_dir = os.path.join(home_dir, "textclassification_results")
logs_dir = os.path.join(home_dir, "textclassification_logs")
model_dir = os.path.join(home_dir, "code-detector-model")

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

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

# Select a single GPU (GPU 3) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if torch.cuda.is_available():
    gpu_id = 0  # This will be GPU 3 because of CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
else:
    print("No GPU available, using CPU")

# 5. Load model
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    logging_dir=logs_dir,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # Explicitly disable data parallelism
    no_cuda=False,
    dataloader_num_workers=4
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
trainer.save_model(model_dir)
