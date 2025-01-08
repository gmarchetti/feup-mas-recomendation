import numpy
import logging

from rankers.ranker_base import RankerBase
from .news_eval.user_history_topic_eval import UserHistoryTopicEval

class HighestPref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)

    def eval_news(self, news, user_topics):
        cand_news_topic = self._data_loader.get_news_topics(news)[0]
        if cand_news_topic in user_topics.keys():
            return user_topics[cand_news_topic]
        else:
            return 0

    def predict(self, impression_group):
        user_history_eval = UserHistoryTopicEval(impression_group)
        news_scores = user_history_eval.score_news_group(impression_group)

        highest_news = news_scores.argmax()

        self.__logger.debug(f"Highest scored news index: {highest_news}")

        news_predictions = numpy.zeros(len(news_scores))

        news_predictions[highest_news] = 1

        return news_predictions