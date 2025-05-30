from transformers import pipeline
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model
clf = pipeline("text-classification", model="code-detector-model", tokenizer="microsoft/codebert-base")

# Load test data
with open("dataset/generated_test.json", "r") as f:
    data = json.load(f)

# Label mapping
label_map = {"LABEL_0": "noCode", "LABEL_1": "containsCode"}
reverse_map = {"noCode": 0, "containsCode": 1}

# Make predictions
true_labels = []
pred_labels = []

print("Evaluating...")

for item in data:
    text = item["text"]
    true = reverse_map[item["label"]]
    prediction = clf(text)[0]["label"]
    pred = 0 if prediction == "LABEL_0" else 1
    
    true_labels.append(true)
    pred_labels.append(pred)

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["noCode", "containsCode"]))

# Plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["noCode", "containsCode"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
