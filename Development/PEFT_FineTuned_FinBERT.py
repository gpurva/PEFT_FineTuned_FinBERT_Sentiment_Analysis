# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 11:20:37 2025

@author: gpurv
"""
#Importing Library
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType

#Moving to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading dataset
data = pd.read_csv("C:\Gaurav\Projects\Financial_News_Sentiment_Analysis\data\data.csv")

#Map Labels
labelid = {'positive':0, 'negative':1, 'neutral':2}
data['Sentiment'] = data["Sentiment"].map(labelid)

#Split data in train, validation and test
#First split in train and temp and then temp into eval and test data
train_df, temp_df = train_test_split(data, stratify = data['Sentiment'], test_size = 0.4, random_state = 42)
eval_df, test_df = train_test_split(temp_df, stratify = temp_df["Sentiment"], test_size = 0.5, random_state = 42)

#Convert ot hugging face dataset
train_df = Dataset.from_pandas(train_df)
eval_df = Dataset.from_pandas(eval_df)
test_df = Dataset.from_pandas(test_df)

#Loading Tokenizer and Model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
base_model = base_model.to(device)

#Defining tokenization function
def tokenize(data):
    return tokenizer(data["Sentence"], truncation = True, padding = True, max_length = 128)

#Tokenize using map
train_df = train_df.map(tokenize, batched = True)
eval_df = eval_df.map(tokenize, batched = True)
test_df = test_df.map(tokenize, batched = True)

#Rename Sentiment column to label as trainer expects label column during training
train_df = train_df.rename_column("Sentiment","labels")
eval_df = eval_df.rename_column("Sentiment","labels")
test_df = test_df.rename_column("Sentiment","labels")

#Set format to pytorch
train_df.set_format(type = "torch", columns = ["input_ids","attention_mask","labels"])
eval_df.set_format(type = "torch", columns = ["input_ids","attention_mask","labels"])
test_df.set_format(type = "torch", columns = ["input_ids","attention_mask","labels"])

#Applying peft lora
peft_config = LoraConfig(task_type = TaskType.SEQ_CLS, r=8, lora_alpha = 16, lora_dropout = 0.1)
model = get_peft_model(base_model, peft_config)
model = model.to(device)

#Define compute metrics
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return {'accuracy': accuracy_score(labels,preds)}

#Trainer Arguments
training_args = TrainingArguments(output_dir = "C:/Gaurav/Projects/Financial_News_Sentiment_Analysis/finbert_lora_output",
                                  learning_rate = 5e-4,
                                  num_train_epochs = 6,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size = 8,
                                  logging_strategy="epoch",
                                  label_names=["labels"])

#Trainer
trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset = train_df,
                  eval_dataset = eval_df,
                  tokenizer = tokenizer,
                  compute_metrics = compute_metrics)

# Train
trainer.train()

# Test Data performance
preds = trainer.predict(test_df)
print(classification_report(test_df['labels'], torch.argmax(torch.tensor(preds.predictions), dim=1),
                            target_names=["positive", "negative", "neutral"]))