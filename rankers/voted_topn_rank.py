import math
import numpy
import logging

from rankers.ranker_base import RankerBase
from rankers.news_eval.trained_eval import TrainedEval

from .news_eval.user_history_topic_eval import UserHistoryTopicEval

class VotedTopNPref(RankerBase):

    def __init__(self):
        super.__init__()
        self.__title_eval = TrainedEval("models/news-prediction/checkpoint-61485")

        self.logger.setLevel(logging.INFO)

        self.__rank_threshold = 1

    def add_votes(votes, news_idx, tallied_votes: dict):
        if news_idx in tallied_votes.keys():
            tallied_votes[news_idx] += votes
        else:
            tallied_votes[news_idx] = votes
    
    def set_rank_threshold(self, threshold):
        self.__rank_threshold = threshold

    def predict(self, impression_info):
        self.__subject_eval = UserHistoryTopicEval(impression_info)

        impression_group = numpy.array(impression_info["candidate_news_index"])
        
        self.logger.debug(f"Candidate news indexes: {impression_group}")
        title_order_preference = self.__title_eval.order_news(impression_info)
        self.logger.debug(f"News ordered by title preference: {title_order_preference}")

        subject_order_preference = self.__subject_eval.order_news(impression_info)
        self.logger.debug(f"News ordered by subject preference: {subject_order_preference}")
        
        news_votes_totals = {}

        for idx in range(len(impression_group)):
            VotedTopNPref.add_votes(len(impression_group) - idx, title_order_preference[idx], news_votes_totals)
            VotedTopNPref.add_votes(len(impression_group) - idx, subject_order_preference[idx], news_votes_totals)

        self.logger.debug("Tallied news votes")
        self.logger.debug(news_votes_totals)
         
        ordered_news = sorted(news_votes_totals, key=lambda x: news_votes_totals.get(x,0), reverse=True)
        self.logger.debug(f"Sorted Candidate news indexes: {ordered_news}")

        news_predictions = []
        for idx in range(len(impression_group)):
            news_predictions.append(0)
            news_position = ordered_news.index(impression_group[idx])
            self.logger.debug(f"News {impression_group[idx]} is the {news_position}-th placed news")
            if news_position < self.__rank_threshold:
                news_predictions[idx] = 1

        self.logger.debug(f"Impression group prediction: {news_predictions}")
        return news_predictions