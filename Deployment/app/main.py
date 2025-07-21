# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 23:01:59 2025

@author: gpurv
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class TextInput(BaseModel):
    text: str

app = FastAPI()

model_path = "app/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.post("/predict")
def predict(input: TextInput):
    tokens = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**tokens).logits
    prediction = torch.argmax(logits, dim=1).item()
    labels = {0: "positive", 1: "negative", 2: "neutral"}
    return {"label": labels[prediction]}