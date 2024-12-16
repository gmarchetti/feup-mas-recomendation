import evaluate
import numpy as np
import os

from datasets import load_dataset
from mind_dataset_loader import MindDatasetFactory
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    full_concat = ""
    for title in examples["hist_titles"]:
        full_concat += title.lower() + "[SEP]"
    
    full_concat += examples["cand_title"].lower()

    return tokenizer(full_concat, truncation=True)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

data_path = f"./data/demo"
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')

data_loader = MindDatasetFactory(train_news_file, train_behaviors_file, 0.05)
mind = data_loader.create_dataset_object()

tokenized_mind = mind.map(preprocess_function, batched=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

id2label = {0: "0", 1: "1"}
label2id = {"0": 0, "1": 1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="models/news-prediction",
    learning_rate=2e-5,
    per_device_train_batch_size=42,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mind,
    eval_dataset=tokenized_mind,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()