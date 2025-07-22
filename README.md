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

## Model Results on test sample
Models	Sentiment	Precision	Recall	F1-Score	# Samples	Accuracy
FinBERT	Positive	0.2	0.38	0.26	371	20%
	Negative	0.02	0.02	0.02	172	
	Neutral	0.43	0.14	0.22	626	
Fine Tune FinBERT	Positive	0.86	0.85	0.85	371	79%
	Negative	0.46	0.45	0.45	172	
	Neutral	0.83	0.84	0.84	626	
<img width="561" height="172" alt="image" src="https://github.com/user-attachments/assets/4bd64bdb-f1ea-4594-a3ba-4b43c89f749c" />
