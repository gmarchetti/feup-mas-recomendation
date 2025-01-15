import logging
import recommenders.datasets.mind as mind
import os

from rankers.ranker_base import RankerBase
from rankers.simple_pref_rank import SimplePref
from rankers.highest_pref_rank import HighestPref
from rankers.voted_topn_rank import VotedTopNPref
from rankers.trained_group_ranker import TrainedGroupRanker
from rankers.news_eval.trained_eval import TrainedEval

from tqdm import tqdm
from models.mind_dataset_loader import MindDatasetFactory
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.models.deeprec.deeprec_utils import cal_metric


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIND_TYPE = 'demo'

data_path = f"./data/{MIND_TYPE}"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_TYPE)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

epochs = 1
seed = 42
batch_size = 1

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)

mind_dataset = MindDatasetFactory(valid_news_file, valid_behaviors_file)

def build_validation_labels_list(size : int = None):
    validation_base_labels = []

    if size is None:
        for impression_data in mind_dataset.get_all_news_offers():
            validation_base_labels.append(impression_data["labels"])
    else:
        for idx in range(size):
            impression_data = mind_dataset.get_news_offer_with_history(idx)
            validation_base_labels.append(impression_data["labels"])

    return validation_base_labels


def evaluate_standalone_models( base_labels : list ):
    ranker_models = [
        SimplePref, 
        HighestPref,
        TrainedGroupRanker,
    ]
    rankers = {}
    predictions = {}

    for ranker_model in ranker_models:
        logger.info(f">>>>> Running {ranker_model.__name__} Rank Model<<<<<")
        rankers[ranker_model.__name__] = ranker_model()
        predictions[ranker_model.__name__] = []
            
        for idx in tqdm(range(len(base_labels))):
            impression_data = mind_dataset.get_news_offer_with_history(idx)
            predicted_labels = rankers[ranker_model.__name__].predict(impression_data)
            predictions[ranker_model.__name__].append(predicted_labels)
            logger.debug(f">> Base:")
            logger.debug(f"{impression_data["labels"]}")
            logger.debug(f">> Predictions")
            logger.debug(f"{predicted_labels}")
            
        logger.info("Evaluating Results")
        logger.info(cal_metric(base_labels, predictions[ranker_model.__name__], ["group_auc", "mean_mrr", "ndcg@5;10"]))

def evaluate_voted_model(base_labels : list):
    ranker = VotedTopNPref()

    for i in range(1,8):
        logger.info(f">>>>> Running Voted Rank Model with {i} Top Ranked News<<<<<")
        
        predictions = []
        ranker.set_rank_threshold(i)
        
        for idx in tqdm(range(len(base_labels))):
            impression_data = mind_dataset.get_news_offer_with_history(idx)
            predicted_labels = ranker.predict(impression_data)
            predictions.append(predicted_labels)
            logger.debug(f">> Base:")
            logger.debug(f"{impression_data["labels"]}")
            logger.debug(f">> Predictions")
            logger.debug(f"{predicted_labels}")
        
        logger.info("Evaluating Results")
        logger.info(cal_metric(base_labels, predictions, ["group_auc", "mean_mrr", "ndcg@5;10"]))

if __name__ == '__main__':
    validation_set = build_validation_labels_list()
    evaluate_standalone_models(validation_set)
    evaluate_voted_model(validation_set)
    