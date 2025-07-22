# FinBERT Sentiment Analysis with LoRA Fine-Tuning (PEFT)

## Project Goal
This project demonstrates fine-tuning of the FinBERT model using Low-Rank Adaptation (LoRA), a Parameter-Efficient Fine-Tuning (PEFT) method, to improve sentiment classification of financial news headlines.

## Dataset
The dataset contains financial news headlines labeled with sentiment:
-**Sentiment labels**: "positive", "negative", "neutral"
-Sample size: 5842 (60% used for training, 20% for evaluation and 20% for testing)
-Source: Kaggle

## Methodology

-**Base model**: `yiyanghkust/finbert-tone`
-**Fine-tuning strategy**: LoRA (Low-Rank Adaptation) using PEFT
-**Library stack**:
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
