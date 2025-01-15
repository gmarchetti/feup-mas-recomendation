import numpy
import logging

from rankers.ranker_base import RankerBase
from .news_eval.user_history_topic_eval import UserHistoryTopicEval

class HighestPref(RankerBase):

    def __init__(self):
        super().__init__()
        self.logger.setLevel(logging.INFO)

    def predict(self, impression_group):
        user_history_eval = UserHistoryTopicEval(impression_group)
        news_scores = user_history_eval.score_news_group(impression_group)

        highest_news = news_scores.argmax()

        self.logger.debug(f"Highest scored news index: {highest_news}")

        news_predictions = numpy.zeros(len(news_scores), dtype="int")

        news_predictions[highest_news] = 1

        return news_predictions