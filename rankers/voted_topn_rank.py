import math
import numpy
import logging

from rankers.ranker_base import RankerBase
from rankers.news_eval.trained_eval import TrainedEval

from .news_eval.user_history_topic_eval import UserHistoryTopicEval

TOP_RANKS = 3
VOTE_THRESHOLD = 3

class VotedTopNPref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)

        self.__title_eval = TrainedEval("models/news-prediction/checkpoint-61485")
        # self.__subject_eval = BaseEval()        

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)

    def add_votes(votes, news_idx, talied_votes: dict):
        if news_idx in talied_votes.keys():
            talied_votes[news_idx] += votes
        else:
            talied_votes[news_idx] = votes

    def predict(self, impression_info):
        self.__subject_eval = UserHistoryTopicEval(impression_info)

        impression_group = numpy.array(impression_info["candidate_news_index"])
        
        self.__logger.debug(f"Candidate news indexes: {impression_group}")
        title_order_preference = self.__title_eval.order_news(impression_info)
        self.__logger.debug(f"News ordered by title preference: {title_order_preference}")

        subject_order_preference = self.__subject_eval.order_news(impression_info)
        self.__logger.debug(f"News ordered by subject preference: {subject_order_preference}")
        
        news_votes_totals = {}

        for idx in range(min(TOP_RANKS, len(impression_group))):
            VotedTopNPref.add_votes(4/(idx+1), title_order_preference[idx], news_votes_totals)
            VotedTopNPref.add_votes(4/(idx+1), subject_order_preference[idx], news_votes_totals)

        self.__logger.debug("Talied news votes")
        self.__logger.debug(news_votes_totals)
        
        news_predictions = []
        for idx in range(len(impression_group)):
            news_predictions.append(0)
            news_votes = news_votes_totals.get(impression_group[idx], 0)
            self.__logger.debug(f"News {impression_group[idx]} got {news_votes} votes")
            if news_votes > VOTE_THRESHOLD:
                news_predictions[idx] = 1

        self.__logger.debug(f"Impression group prediction: {news_predictions}")
        return news_predictions