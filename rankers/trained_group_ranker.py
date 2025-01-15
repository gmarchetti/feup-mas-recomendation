import math
import numpy
import logging

from rankers.ranker_base import RankerBase
from rankers.news_eval.base_eval import BaseEval
from rankers.news_eval.trained_eval import TrainedEval

TOP_RANKS = 2

class TrainedGroupRanker(RankerBase):

    def __init__(self):
        super().__init__()

        self.__title_eval = TrainedEval("models/news-prediction/checkpoint-61485")
        self.__logger = logging.getLogger(__name__)

    def predict(self, impression_info):

        news_predictions = self.__title_eval.predict_impression_group_labels(impression_info)

        return news_predictions