import logging
import recommenders.datasets.mind as mind
import os

from rankers.ranker_base import RankerBase
from recommenders.news_feed_from_training import build_news_feed
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.models.deeprec.deeprec_utils import cal_metric
# from recommenders.utils.notebook_utils import store_metadata

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MIND_type = 'demo'

def random_eval():
    user_sessions, user_click_history = mind.read_clickhistory("train", "behaviors.tsv")
    logger.info(">>>> User Sessions <<<<")
    ranker = RankerBase()
    
    for idx in range(0, 4):
        user_session = user_sessions[idx]
        user_id = user_session[0]
        
        logger.info(f"News offering for user {user_id}")
        user_news_feed = build_news_feed(user_session)
        logger.info(build_news_feed(user_session))

        logger.info(">>> Ranks for news offering")
        logger.info(ranker.eval(user_news_feed, user_id))

# if __name__ == '__main__':
#     random_eval()

data_path = f"./data/{MIND_type}"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

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
batch_size = 32

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)

iterator = MINDIterator
# model = Model(hparams, iterator, seed=seed)
print(cal_metric([[0,1,0,0],[1,0,0,0]], [[0,1,0,0],[1,0,0,0]], hparams.metrics))
# print(model.run_eval(valid_news_file, valid_behaviors_file))