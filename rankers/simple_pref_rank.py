import logging

from rankers.ranker_base import RankerBase
from .news_eval.user_history_topic_eval import UserHistoryTopicEval

class SimplePref(RankerBase):

    def __init__(self):
        super().__init__()
        
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)

    def predict(self, impression_group):
        user_history_eval = UserHistoryTopicEval(impression_group)
        news_scores = user_history_eval.score_news_group(impression_group)

        news_predictions = []

        for score in news_scores:
            if score > 0:
                news_predictions.append(1)
            else:
                news_predictions.append(0)

        return news_predictions