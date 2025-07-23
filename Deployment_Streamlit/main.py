
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "C:\Gaurav\Projects\Financial_News_Sentiment_Analysis\Deployment_Streamlit\model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label mapping
labels = {0: "Positive", 1: "Negative", 2: "Neutral"}

# Streamlit UI
st.title("Fine Tuned FinBERT Sentiment Analysis")
text = st.text_area("Enter financial news:", height=300)

if st.button("Run Model and Get News Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**tokens).logits
            prediction = torch.argmax(logits, dim=1).item()
            st.success(f"**Sentiment:** {labels[prediction]}")
