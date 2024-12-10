from rankers.ranker_base import RankerBase

class HighestPref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)

    def eval_news(self, news, user_topics):
        cand_news_topic = self._data_loader.get_news_topics(news)[0]
        if cand_news_topic in user_topics.keys():
            return user_topics[cand_news_topic]
        else:
            return 0
    
    def predict(self, impression_group, impr_index):
        user_history = self._data_loader.get_user_history(impr_index)
        user_topics = {}

        for news_hist in user_history:
            news_topic = self._data_loader.get_news_topics(news_hist)[0]
            if news_topic not in user_topics.keys():
                user_topics[news_topic] = 1
            else:
                user_topics[news_topic] += 1

        news_predictions = []
        
        highest_score = -1
        highest_score_index = -1
        idx = 0

        for news_cand in impression_group:
            news_score = self.eval_news(news_cand, user_topics)
            
            if news_score > highest_score:
                highest_score = news_score
                highest_score_index = idx
            
            idx += 1

        for idx in range(len(impression_group)):
            news_predictions.append(0)
            if idx == highest_score_index:
                news_predictions[idx] = 1

        return news_predictions