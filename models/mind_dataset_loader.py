# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import os

from tqdm import tqdm
from datasets import IterableDataset, Dataset

def dummy_gen():
    yield {"title" : "this works"}

class MindDatasetFactory():
    __impr_indexes = []
    __news_topics_by_index = []
    __title_topics_by_index = []
    __histories = []
    __imprs = []
    __labels = []


    def dummy_gen_2():
        yield {"title" : "and this works too"}

    def generate_training_sample():
        for impr_idx in MindDatasetFactory.__impr_indexes:
            history_titles = []
            for hist_news in MindDatasetFactory.__histories[impr_idx]:
                history_titles.append(MindDatasetFactory.__title_topics_by_index[hist_news])
            
            candidate_impressions = MindDatasetFactory.__imprs[impr_idx]
            candidate_labels = MindDatasetFactory.__labels[impr_idx]
            for cand_news_index in range(len(candidate_impressions)):
                cand_news_title = MindDatasetFactory.__title_topics_by_index[candidate_impressions[cand_news_index]]
                cand_label = candidate_labels[cand_news_index]
                training_vect = {"hist_titles" : history_titles, "cand_title" : cand_news_title, "label" : cand_label}
                yield training_vect

    def create_dataset_object(self):
        mind = Dataset.from_generator(MindDatasetFactory.generate_training_sample)
        return mind


    def __init__(self, news_file, behavior_file, col_spliter="\t", ID_spliter="%"):

        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        print("Reading News File")
        self.__init_news(news_file)
        print("Reading Behavior File")
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
                MindDatasetFactory.__title_topics_by_index.append(title)

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
                impr_news = [self.__nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]

                MindDatasetFactory.__histories.append(history)
                MindDatasetFactory.__imprs.append(impr_news)
                MindDatasetFactory.__labels.append(label)
                MindDatasetFactory.__impr_indexes.append(impr_index)

                impr_index += 1



if __name__ == '__main__':
    data_path = f"./data/demo"

    train_news_file = os.path.join(data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')

    data_loader = MindDatasetFactory(train_news_file, train_behaviors_file)

    mind = data_loader.create_dataset_object()

    for vector in tqdm(mind):
        print(vector)

    
