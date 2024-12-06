import logging
import dataset_tools.mind as mind

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
user_sessions, user_click_history = mind.read_clickhistory("train", "behaviors.tsv")

if __name__ == '__main__':
    logger.info(">>>> User Sessions <<<<")
    for idx in range(0, 4):
        user_session = user_sessions[idx]
        logger.info(user_session)
        logger.info(">>< User id")
        logger.info(user_session[0])
        logger.info(">>< Clicks")
        logger.info(user_session[1])
        logger.info(">>< Positives")
        logger.info(user_session[2])
        logger.info(">>< Negatives")
        logger.info(user_session[3])
        user_id = user_session[0]
        logger.info(f"User {user_id} click history")
        logger.info(user_click_history[user_id])

       