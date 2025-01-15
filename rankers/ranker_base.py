import random
import logging

class RankerBase():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict(self, impression_group, impr_index):
        return []