import numpy
import logging
from rankers.ranker_base import RankerBase

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
        user_history = impression_group["history_topics"]
        self.__logger.debug(f"User history topics: {user_history}")
        
        topic_scores = {}
        
        for history_topic_group in user_history:
            for topic in history_topic_group:
                current_topic_score = topic_scores.get(topic, 0)
                topic_scores[topic] = current_topic_score + 1

        self.__logger.debug(f"User history scored topics: {topic_scores}")

        cand_news_topics_group : list = impression_group["candidates_topics"] 

        news_scores = numpy.array([])
        
        for cand_topics in cand_news_topics_group:
            news_score = topic_scores.get(cand_topics[0], 0) + topic_scores.get(cand_topics[1], 0)
            news_scores = numpy.append(news_scores, news_score)
        
        self.__logger.debug(f"Candidate news scores: {news_scores}")

        highest_news = news_scores.argmax()

        self.__logger.debug(f"Highest scored news index: {highest_news}")

        news_predictions = numpy.zeros(len(news_scores))

        news_predictions[highest_news] = 1

        return news_predictions