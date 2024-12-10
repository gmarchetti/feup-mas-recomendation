from rankers.ranker_base import RankerBase

class SimplePref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)


    def predict(self, impression_group, impr_index):
        user_history = self._data_loader.get_user_history(impr_index)
        user_topics = []

        for news_hist in user_history:
            news_topic = self._data_loader.get_news_topics(news_hist)[0]
            if news_topic not in user_topics:
                user_topics.append(news_topic)

        news_predictions = []
        for news_cand in impression_group:
            cand_news_topic = self._data_loader.get_news_topics(news_cand)[0]

            if cand_news_topic in user_topics:
                news_predictions.append(1)
            else:
                news_predictions.append(0)

        return news_predictions