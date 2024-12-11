import evaluate
import numpy as np
import os
import torch

from datasets import load_dataset
from models.mind_dataset_loader import MindDatasetFactory
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("models/news_prediction/checkpoint-73")
model = AutoModelForSequenceClassification.from_pretrained("models/news_prediction/checkpoint-73", num_labels=2)
accuracy = evaluate.load("accuracy")

def preprocess_function(examples):
    full_concat = "; ".join(examples["hist_titles"]) + '; ' +examples["cand_title"]
    return tokenizer(full_concat, truncation=True, return_tensors="pt")

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

data_path = f"./data/demo"
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')

data_loader = MindDatasetFactory(train_news_file, train_behaviors_file, under_sampling_rate=0.01)

mind = data_loader.create_dataset_object()

# mini_mind = mind.shard(num_shards=100, index=0)

with torch.no_grad():
    for sample_sentence in mind:
        # sample_sentence = mind[idx]
        print(sample_sentence)

        tokenized_sample = preprocess_function(sample_sentence)
        logits = model(**tokenized_sample).logits

        predicted_class_id = logits.argmax().item()
        print(model.config.id2label[predicted_class_id])

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}

