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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

#Moving to GPU
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

#Loading Tokenizer and Model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
base_model = base_model.to(device)

########## Checking performance with base model ################################
test_df_base = test_df

t_input = tokenizer(list(test_df["Sentence"]), truncation = True, padding=True, return_tensors='pt')

#Convert labels to tensor
t_label = torch.tensor(test_df["Sentiment"].tolist())

input_ids = t_input["input_ids"]
attention_mask = t_input["attention_mask"]

# Create DataLoader for batching
dataset = TensorDataset(input_ids, attention_mask, t_label)
loader = DataLoader(dataset, batch_size=32)

# Store predictions and true labels
all_preds = []
all_labels = []

#Put model to evaluation mode. By default model is in trainable mode. Layers  like dropout and batchnorm behaves differently during training and evaluation. Putting model to evaluation mode ensures stable and deterministic output during evaluation
base_model.eval()

#Performing prediction. The no_grad will disable gradient traction which is only useful during training and not during evaluation
with torch.no_grad():
    for batch in tqdm(loader):
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]

        output = base_model(input_ids=b_input_ids, attention_mask=b_mask)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=["positive","negative","neutral"]))

#                precision    recall  f1-score   support
#    positive       0.20      0.38      0.26       371
#    negative       0.02      0.02      0.02       172
#     neutral       0.43      0.14      0.22       626

#    accuracy                           0.20      1169
#   macro avg       0.21      0.18      0.16      1169
#weighted avg       0.29      0.20      0.20      1169

################################# Fine Tuning Base Model ############################
#Convert ot hugging face dataset
train_df = Dataset.from_pandas(train_df)
eval_df = Dataset.from_pandas(eval_df)
test_df = Dataset.from_pandas(test_df)

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
peft_config = LoraConfig(r=8, lora_alpha = 16, target_modules=["query", "value"], lora_dropout = 0.1, bias="none", task_type = "SEQ_CLS")
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

#                precision    recall  f1-score   support
#    positive       0.86      0.85      0.85       371
#    negative       0.46      0.45      0.45       172
#     neutral       0.83      0.84      0.84       626

#    accuracy                           0.79      1169
#   macro avg       0.72      0.71      0.71      1169
# weighted avg       0.78      0.79      0.78      1169
