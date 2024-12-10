import random
import logging
from rankers.io.news_data_iterator import NewsDataIterator

class RankerBase():
    def __init__(self, iterator: NewsDataIterator):
        self.__logger = logging.getLogger(__name__)
        self._data_loader = iterator

    def predict(self, impression_group, impr_index):
        predictions = []
        for i in range(len(impression_group)):
            predictions.append(random.randint(0, 1))
        return predictions