import evaluate
import numpy as np
import os
import logging
import json

from pickle import dump
from tqdm import tqdm

from models.mind_dataset_loader import MindDatasetFactory
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MIND_TYPE = 'small'

data_path = f"./data/{MIND_TYPE}"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_TYPE)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)

data_loader = MindDatasetFactory(train_news_file, train_behaviors_file, 0.15)
feature_dict, label_vector = data_loader.create_dataset_for_sklearn()


vectorizer = DictVectorizer()

logger.info("Vectorizing features")
feature_vec = vectorizer.fit_transform(feature_dict).toarray()

gnb = MultinomialNB()
logger.info("Training bayesian classifier")

for idx in tqdm(range(len(feature_vec))):
    gnb.partial_fit([feature_vec[idx]], [label_vector[idx]], [0, 1])

with open("models/vector.pkl", "wb") as f:
    dump(vectorizer, f, protocol=5)

with open("models/gnb.pkl", "wb") as f:
    dump(gnb, f, protocol=5)
