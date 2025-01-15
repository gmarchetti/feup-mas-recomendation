import random
import logging
from rankers.io.news_data_iterator import NewsDataIterator

class RankerBase():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict(self, impression_group, impr_index):
        return []