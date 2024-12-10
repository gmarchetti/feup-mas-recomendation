import math
import numpy

from rankers.ranker_base import RankerBase
from rankers.news_eval.base_eval import BaseEval

class NegotiatePref(RankerBase):

    def __init__(self, iterator):
        super().__init__(iterator)

        self.__title_eval = BaseEval()
        self.__subject_eval = BaseEval()        

        self.__eps = 0.2

    
    def predict(self, impression_group, impr_index):

        round = 0

        title_value = self.__title_eval.eval_news_offer(impression_group, round)
        subject_value = self.__subject_eval.eval_news_offer(impression_group, round)
        
        predicted_news_order = numpy.copy(impression_group)
        
        while abs(title_value - subject_value) > self.__eps:
            if round > 10:
                # print("Too many round, giving up")
                break
            
            # offer
            predicted_news_order = self.__title_eval.order_news(impression_group)

            title_value = self.__title_eval.eval_news_offer(predicted_news_order, round)
            subject_value = self.__subject_eval.eval_news_offer(predicted_news_order, round)

            if abs(title_value - subject_value) < self.__eps:
                # print("Deal reached")
                break

            #counter offer
            predicted_news_order = self.__subject_eval.order_news(predicted_news_order)

            title_value = self.__title_eval.eval_news_offer(predicted_news_order, round)
            subject_value = self.__subject_eval.eval_news_offer(predicted_news_order, round)

        # print("Initial order of news:", impression_group)
        # print("Final order of news: ", predicted_news_order)
        highest_ranked_news = predicted_news_order[0]
        highest_ranked_news_index = numpy.where(impression_group == highest_ranked_news)[0][0]
        # print("Highest ranked news index:", highest_ranked_news_index)
        news_predictions = []
        for idx in range(len(impression_group)):
            news_predictions.append(0)
            if idx == highest_ranked_news_index:
                news_predictions[idx] = 1

        return news_predictions