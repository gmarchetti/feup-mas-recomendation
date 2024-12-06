import random

class RankerBase():
    def eval(self, news: list, user_id: str):
        ranking = list(range(1, len(news) + 1))

        random.shuffle(ranking)
        return ranking