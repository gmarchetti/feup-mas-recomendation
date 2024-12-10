import math
import numpy

from rankers.ranker_base import RankerBase
from rankers.news_eval.base_eval import BaseEval

class VotedPref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)

        self.__title_eval = BaseEval()
        self.__subject_eval = BaseEval()        

        self.__eps = 0.2

    def add_votes(votes, news_idx, talied_votes: dict):
        if news_idx in talied_votes.keys():
            talied_votes[news_idx] += votes
        else:
            talied_votes[news_idx] = votes

    def predict(self, impression_group, impr_index):

        round = 0

        title_order_preference = self.__title_eval.order_news(impression_group)
        subject_order_preference = self.__subject_eval.order_news(impression_group)
        
        news_votes = {}

        for idx in range(min(2, len(impression_group))):
            VotedPref.add_votes(4/(idx+1), title_order_preference[idx], news_votes)
            VotedPref.add_votes(4/(idx+1), subject_order_preference[idx], news_votes)

        print("Talied news votes")
        print(news_votes)
        # print("Initial order of news:", impression_group)
        # print("Final order of news: ", predicted_news_order)
        total_winner_votes = max(news_votes.values())
        print("Winners vote: ", total_winner_votes)      
        highest_ranked_news = [key for key in news_votes if news_votes[key] == total_winner_votes]
        print("Highest ranked news", highest_ranked_news)
        
        highest_ranked_news_index = []
        for news in highest_ranked_news:
            highest_ranked_news_index.append(numpy.where(impression_group == news)[0][0])
        
        print("Highest ranked news index:", highest_ranked_news_index)
        news_predictions = []
        for idx in range(len(impression_group)):
            news_predictions.append(0)
            if idx in highest_ranked_news_index:
                news_predictions[idx] = 1

        return news_predictions