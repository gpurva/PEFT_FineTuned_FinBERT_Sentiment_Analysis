# FinBERT Sentiment Analysis with LoRA Fine-Tuning (PEFT)

## Project Goal
This project demonstrates fine-tuning of the FinBERT model using Low-Rank Adaptation (LoRA), a Parameter-Efficient Fine-Tuning (PEFT) method, to improve sentiment classification of financial news headlines.

## ## Live Demo
The model is live on Streamlit. Check out the llive app here: https://peftfinetunedfinbertsentimentanalysis-gauravpurva.streamlit.app/

## Dataset
The dataset contains financial news headlines labeled with sentiment:

a. **Sentiment labels**: "positive", "negative", "neutral"

b. Sample size: 5842 (60% used for training, 20% for evaluation and 20% for testing)

c. Source: Kaggle

## Methodology
a. **Base model**: `yiyanghkust/finbert-tone`

b. **Fine-tuning strategy**: LoRA (Low-Rank Adaptation) using PEFT

c. **Library stack**:
  1. transformers
  2. torch
  3. peft
  4. scikit-learn
  5. pandas 


## LoRA Config
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

## Model Results on test sample

<img width="561" height="172" alt="image" src="https://github.com/user-attachments/assets/8134e94d-5a3f-4c25-ac26-dad69a53682c" />

