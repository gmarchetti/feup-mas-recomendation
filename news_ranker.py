import logging
import recommenders.mind as mind

from rankers.ranker_base import RankerBase
from recommenders.news_feed_from_training import build_news_feed

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

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

if __name__ == '__main__':
    random_eval()



        

       