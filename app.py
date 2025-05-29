from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import torch
import uvicorn

# Load model and tokenizer
model_path = "code-detector-model"
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Create pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Init FastAPI
app = FastAPI(title="Code Detector API")

# Define request model
class TextInput(BaseModel):
    text: str

# Health check
@app.get("/")
def read_root():
    return {"status": "running"}

# Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    try:
        result = classifier(input.text)[0]
        label = result['label']
        score = round(result['score'], 4)
        label_map = {"LABEL_0": "noCode", "LABEL_1": "containsCode"}
        return {
            "text": input.text,
            "prediction": label_map[label],
            "confidence": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8060, reload=True)