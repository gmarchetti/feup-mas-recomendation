import random

class RankerBase():
    def eval(self, news: list):
        return random.uniform(1, len(news))