import logging
from rankers.ranker_base import RankerBase

class SimplePref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)
        
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)

    def predict(self, impression_group):
        user_history = impression_group["history_topics"]
        self.__logger.debug(f"User history topics: {user_history}")
        
        user_topics = []

        for history_topic_group in user_history:
            for history_topic in history_topic_group:
                if not history_topic in user_topics:
                    user_topics.append(history_topic)

        self.__logger.debug(f"Flattened user history: {user_topics}")
        news_predictions = []

        cand_news_topics : list = impression_group["candidates_topics"] 

        for news_cand in cand_news_topics:
            self.__logger.debug(f"Checking if candidate news topics {news_cand} are on user history")
            
            has_topics = False

            for topic in news_cand:
                has_topics = has_topics or (topic in user_topics)

            if has_topics:
                self.__logger.debug(f"Matches user history, predicting 1")
                news_predictions.append(1)
            else:
                self.__logger.debug(f"Does not matches user history, predicting 0")
                news_predictions.append(0)

        return news_predictions