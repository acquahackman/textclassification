from transformers import pipeline

clf = pipeline("text-classification", model="code-detector-model", tokenizer="microsoft/codebert-base")
print(clf("This is just a test."))
print(clf("def add(x, y): return x + y"))
print(clf("I will add some code here:def add(x, y): return x + y"))
