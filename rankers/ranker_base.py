import random
import logging

class RankerBase():
    def __init__(self):
        self.__logger = logging.getLogger(__name__)

    def predict(self, impression_group):
        predictions = []
        for i in range(len(impression_group)):
            predictions.append(random.randint(0, 1))
        return predictions