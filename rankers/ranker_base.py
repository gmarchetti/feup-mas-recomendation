import random
import logging
import keras

from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.nrms import NRMSModel

class RandomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = logging.getLogger(__name__)

    def call(self, input):
        # self.__logger.info(input)
        return [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    
    def predict_on_batch(self, input):
        # self.__logger.info(input)
        return [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]

class RankerBase(NRMSModel):
    def __init__(self, hparams, iterator_creator, seed=None):
        super().__init__(hparams, iterator_creator, seed)

        self.__logger = logging.getLogger(__name__)

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        self.__logger.info(input_feat, input_label)
        return input_feat, input_label
    
    def _build_graph(self):
        model = RandomModel()
        scorer = RandomModel()


        self.newsencoder = model
        self.userencoder = model

        return model, scorer

    def eval(self, news: list, user_id: str):
        ranking = list(range(1, len(news) + 1))

        random.shuffle(ranking)
        return ranking