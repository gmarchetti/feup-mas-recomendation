import logging
import recommenders.datasets.mind as mind
import os

from rankers.ranker_base import RankerBase
from rankers.simple_pref_rank import SimplePref
from rankers.highest_pref_rank import HighestPref
from rankers.negotiate_rank import NegotiatePref
from rankers.voted_rank import VotedPref
from rankers.news_eval.trained_eval import TrainedEval

from tqdm import tqdm
from models.mind_dataset_loader import MindDatasetFactory
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from rankers.io.news_data_iterator import NewsDataIterator
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
base_labels = {}

ranker_models = [
    # RankerBase, 
    # SimplePref, 
    # HighestPref,
    # NegotiatePref,
    VotedPref
    ]
rankers = {}
predictions = {}

for ranker_model in ranker_models:
    rankers[ranker_model.__name__] = ranker_model(mind_dataset)
    predictions[ranker_model.__name__] = []
    base_labels[ranker_model.__name__] = []
# for impr_indexes, impr_news, uindexes, impr_label in tqdm(iterator.load_impression_from_file(valid_behaviors_file)):

# for ranker_model in ranker_models:
    # trained_eval = TrainedEval("models/news-prediction/checkpoint-9414")
    impression_data = mind_dataset.get_news_offer_with_history(0)
    logger.debug(impression_data)
    # trained_eval.order_news(impression_data)
    predicted_labels = rankers[ranker_model.__name__].predict(impression_data)
    predictions[ranker_model.__name__].append(predicted_labels)
    base_labels[ranker_model.__name__].append(impression_data["labels"])
    print(">> Base")
    print(impression_data["labels"])
    print(">> Predictions")
    print(predicted_labels)

# print(iterator.load_data_from_file(valid_news_file, valid_behaviors_file).__next__())

for ranker_model in ranker_models:
    print(f">>>>> {ranker_model.__name__} <<<<<")    
    print(cal_metric(base_labels[ranker_model.__name__], predictions[ranker_model.__name__], hparams.metrics))