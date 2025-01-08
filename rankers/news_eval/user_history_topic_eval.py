import logging
import numpy

class UserHistoryTopicEval():

    def __init__(self, impression_group):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)

        user_history = impression_group["history_topics"]
        self.__logger.debug(f"User history topics: {user_history}")
        
        self.__topic_scores = {}
        
        for history_topic_group in user_history:
            for topic in history_topic_group:
                current_topic_score = self.__topic_scores.get(topic, 0)
                self.__topic_scores[topic] = current_topic_score + 1

        self.__logger.debug(f"User history scored topics: {self.__topic_scores}")
    
    def score_news_group(self, impression_group):
        cand_news_topics_group : list = impression_group["candidates_topics"] 

        news_scores = numpy.array([])
        
        self.__logger.debug(f"Candidate news topics: {cand_news_topics_group}")
        for cand_topics in cand_news_topics_group:
            news_score = self.__topic_scores.get(cand_topics[0], 0) + self.__topic_scores.get(cand_topics[1], 0)
            news_scores = numpy.append(news_scores, news_score)

        self.__logger.debug(f"Candidate news scores: {news_scores}")

        return news_scores