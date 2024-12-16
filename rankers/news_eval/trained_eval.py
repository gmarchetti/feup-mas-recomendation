import random
import math
import numpy
import torch
import logging

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

class TrainedEval():
    
    def preprocess_function(self, hist_titles, cand_title):
        full_concat = ""
        for title in hist_titles:
            full_concat += title.lower() + "[SEP]"
        
        full_concat += cand_title.lower()

        return self.__tokenizer(full_concat, truncation=True, return_tensors="pt")
    
    def __init__(self, model_path):
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.__logger = logging.getLogger(__name__)

    def predict(self, hist_titles, cand_title):
        with torch.no_grad():
            
            tokenized_sample = self.preprocess_function(hist_titles, cand_title)
            logits = self.__model(**tokenized_sample).logits[0][1].item()

            return logits

    def order_news(self, impression_data):
        
        cand_news = impression_data["candidate_news_index"]
        self.__logger.debug(">>> Candidate news indexes")
        self.__logger.debug(cand_news)
        news_values_dict = {}
        idx = 0
        for cand_news_title in impression_data["candidate_titles"]:
            news_values_dict[cand_news[idx]] = self.predict(impression_data["history_titles"], cand_news_title)
            idx += 1
        
        self.__logger.debug(">>> Candidate news values")
        self.__logger.debug(news_values_dict)
        ordered_news = sorted(cand_news, key=lambda x: news_values_dict[x], reverse=True)
        self.__logger.debug(">>> Sorted Candidate news indexes")
        self.__logger.debug(ordered_news)
        return ordered_news
    
    def eval_news_offer(self, news, round):
        return random.uniform(1,4) * len(news) * math.pow(self._gama, round)