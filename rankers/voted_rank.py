import math
import numpy
import logging

from rankers.ranker_base import RankerBase
from rankers.news_eval.base_eval import BaseEval
from rankers.news_eval.trained_eval import TrainedEval

from .news_eval.user_history_topic_eval import UserHistoryTopicEval

TOP_RANKS = 2

class VotedPref(RankerBase):

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
        
        news_votes = {}

        for idx in range(min(TOP_RANKS, len(impression_info))):
            VotedPref.add_votes(4/(idx+1), title_order_preference[idx], news_votes)
            VotedPref.add_votes(4/(idx+1), subject_order_preference[idx], news_votes)

        self.__logger.debug("Talied news votes")
        self.__logger.debug(news_votes)

        total_winner_votes = max(news_votes.values())
        self.__logger.debug(f"Winners vote: {total_winner_votes}")      
        highest_ranked_news = numpy.array([key for key in news_votes if news_votes[key] == total_winner_votes])
        self.__logger.debug(f"Highest ranked news: {highest_ranked_news}")
        
        highest_ranked_news_index = []
        for news in highest_ranked_news:
            highest_ranked_news_index.append(numpy.where(impression_group == news)[0][0])
        
        self.__logger.debug(f"Highest ranked news index in impression: {highest_ranked_news_index}")
        news_predictions = []
        for idx in range(len(impression_group)):
            news_predictions.append(0)
            if idx in highest_ranked_news_index:
                news_predictions[idx] = 1

        return news_predictions