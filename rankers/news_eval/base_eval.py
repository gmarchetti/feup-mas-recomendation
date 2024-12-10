import random
import math
import numpy

class BaseEval():
    def __init__(self):
        self._gama = 0.5

    def order_news(self, news):
        
        ordered_news = numpy.copy(news)
        random.shuffle(ordered_news)
        return ordered_news
    
    def eval_news_offer(self, news, round):
        return random.uniform(1,4) * len(news) * math.pow(self._gama, round)