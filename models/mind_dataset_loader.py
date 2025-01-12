# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import os
import random

from tqdm import tqdm
from datasets import IterableDataset, Dataset

class MindDatasetFactory():
    __impr_indexes = []
    __news_topics_by_index = []
    __news_titles_by_index = []
    __histories = []
    __imprs = []
    __labels = []
    __sampling_rate = 1

    def generate_training_sample():
        for impr_idx in MindDatasetFactory.__impr_indexes:
            history_titles = []
            for hist_news in MindDatasetFactory.__histories[impr_idx]:
                history_titles.append(MindDatasetFactory.__news_titles_by_index[hist_news])
            
            candidate_impressions = MindDatasetFactory.__imprs[impr_idx]
            candidate_labels = MindDatasetFactory.__labels[impr_idx]
            for cand_news_index in range(len(candidate_impressions)):
                cand_news_title = MindDatasetFactory.__news_titles_by_index[candidate_impressions[cand_news_index]]
                cand_label = candidate_labels[cand_news_index]
                training_vect = {"hist_titles" : history_titles, "cand_title" : cand_news_title, "label" : cand_label}
                yield training_vect

    def generate_topic_training_sample():
        for impr_idx in MindDatasetFactory.__impr_indexes:
            history_topics = []
            for hist_news in MindDatasetFactory.__histories[impr_idx]:
                history_topics.append(MindDatasetFactory.__news_topics_by_index[hist_news])
            
            candidate_impressions = MindDatasetFactory.__imprs[impr_idx]
            candidate_labels = MindDatasetFactory.__labels[impr_idx]
            for cand_news_index in range(len(candidate_impressions)):
                cand_news_topics = MindDatasetFactory.__news_topics_by_index[candidate_impressions[cand_news_index]]
                cand_label = candidate_labels[cand_news_index]
                training_vect = {"history_topics" : history_topics, "candidate_topics" : cand_news_topics, "label" : cand_label}
                yield training_vect

    def create_dataset_object(self):
        mind = Dataset.from_generator(MindDatasetFactory.generate_training_sample)
        return mind

    def create_topic_dataset_object(self):
        mind = Dataset.from_generator(MindDatasetFactory.generate_topic_training_sample)
        return mind
    
    def create_dataset_for_sklearn(self):
        feature_vector = []
        label_vector = []
        
        for impr_idx in MindDatasetFactory.__impr_indexes:
            history_topics = []
            for hist_news in MindDatasetFactory.__histories[impr_idx]:
                history_topics.append(MindDatasetFactory.__news_topics_by_index[hist_news][0])
                history_topics.append(MindDatasetFactory.__news_topics_by_index[hist_news][1])
            
            candidate_impressions = MindDatasetFactory.__imprs[impr_idx]
            candidate_labels = MindDatasetFactory.__labels[impr_idx]
            
            for cand_news_index in range(len(candidate_impressions)):
                cand_news_topics = MindDatasetFactory.__news_topics_by_index[candidate_impressions[cand_news_index]]
                cand_label = candidate_labels[cand_news_index]
                feature_vector.append({"history_topics" : history_topics, "candidate_topics" : cand_news_topics})
                label_vector.append(cand_label)

        return feature_vector, label_vector

    def get_news_offer_with_history(self, impr_idx):
        
        history_titles = []
        history_topics = []
        for hist_news in MindDatasetFactory.__histories[impr_idx]:
            history_titles.append(MindDatasetFactory.__news_titles_by_index[hist_news])
            history_topics.append(MindDatasetFactory.__news_topics_by_index[hist_news])
        
        candidate_impressions = MindDatasetFactory.__imprs[impr_idx]
        candidate_titles_array = []
        candidate_labels_array = []
        candidate_topics_array = []
        candidate_labels = MindDatasetFactory.__labels[impr_idx]
        for cand_news_index in range(len(candidate_impressions)):
            candidate_titles_array.append(MindDatasetFactory.__news_titles_by_index[candidate_impressions[cand_news_index]])
            candidate_labels_array.append(candidate_labels[cand_news_index])
            candidate_topics_array.append(MindDatasetFactory.__news_topics_by_index[candidate_impressions[cand_news_index]])
        
        return {"history_titles" : history_titles, 
                "history_topics": history_topics, 
                "candidate_titles" : candidate_titles_array, 
                "candidates_topics" : candidate_topics_array,
                "candidate_news_index" : candidate_impressions,
                "labels" : candidate_labels}        

    def get_all_news_offers(self):
        for impr_idx in MindDatasetFactory.__impr_indexes:
            yield self.get_news_offer_with_history(impr_idx)
            
    def __init__(self, news_file, behavior_file, under_sampling_rate=1, col_spliter="\t", ID_spliter="%"):

        MindDatasetFactory.__sampling_rate = under_sampling_rate
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.__init_news(news_file)
        self.__init_behaviors(behavior_file)

    def __init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.__nid2index = {}
        
        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in tqdm(rd):
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )
                
                if nid in self.__nid2index:
                    continue

                self.__nid2index[nid] = len(self.__nid2index)
                MindDatasetFactory.__news_topics_by_index.append([vert, subvert])
                MindDatasetFactory.__news_titles_by_index.append(title)

    def __init_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in tqdm(rd):
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.__nid2index[i] for i in history.split()]
                impr_news = []
                label = []
                for news in impr.split():
                    news_label = int(news.split("-")[1])
                    news_index = self.__nid2index[news.split("-")[0]]
                    random_draw = random.uniform(0.0, 1.0)
                    # print(random_draw, news_label)
                    if news_label == 1 or  random_draw < MindDatasetFactory.__sampling_rate:
                        # print("Positive label or random select")
                        label.append(news_label)
                        impr_news.append(news_index)
                    # else:
                        # print("Ignoring sample")

                MindDatasetFactory.__histories.append(history)
                MindDatasetFactory.__imprs.append(impr_news)
                MindDatasetFactory.__labels.append(label)
                MindDatasetFactory.__impr_indexes.append(impr_index)

                impr_index += 1



if __name__ == '__main__':
    data_path = f"./data/demo"

    train_news_file = os.path.join(data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')

    data_loader = MindDatasetFactory(train_news_file, train_behaviors_file, 0.01)

    mind = data_loader.create_dataset_object()

    for vector in tqdm(mind):
        print(vector)

    
