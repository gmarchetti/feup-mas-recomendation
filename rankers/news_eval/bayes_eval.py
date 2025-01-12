from pickle import load
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

import logging
import numpy

class BayesTopicEval():

    def __init__(self, _):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)

        with open("models/vector.pkl", "rb") as f:
            self.__vectorizer : DictVectorizer = load(f)

        with open("models/gnb.pkl", "rb") as f:
            self.__gnb : MultinomialNB = load(f)
    
    def __predict_single_news(self, cand_news_topics, hist_news_topics):
        hist_topics = []
        for topic in hist_news_topics:
            hist_topics.append(topic[0])
            hist_topics.append(topic[1])
        
        feature_dict = {"history_topics" : hist_topics, "candidate_topics" : cand_news_topics}        
        self.__logger.debug("Feature Dictionary")
        self.__logger.debug(feature_dict)
        feature_vect = self.__vectorizer.transform(feature_dict).toarray()
        self.__logger.debug("Feature Vector")
        self.__logger.debug(feature_vect)
        prediction = self.__gnb.predict(feature_vect)
        self.__logger.debug("Prediction")
        self.__logger.debug(prediction)
        return prediction[0]

    def __get_candidate_news_score(self, cand_news_topics, hist_news_topics):
        hist_topics = []
        for topic in hist_news_topics:
            hist_topics.append(topic[0])
            hist_topics.append(topic[1])
        
        feature_dict = {"history_topics" : hist_topics, "candidate_topics" : cand_news_topics}        
        self.__logger.debug("Feature Dictionary")
        self.__logger.debug(feature_dict)
        feature_vect = self.__vectorizer.transform(feature_dict).toarray()
        self.__logger.debug("Feature Vector")
        self.__logger.debug(feature_vect)
        prediction = self.__gnb.predict_log_proba(feature_vect)
        self.__logger.debug("Prediction")
        self.__logger.debug(prediction)

        return prediction[0][1]

    def predict(self, impression_data):
        predictions = []
        for cand_topic in impression_data["candidates_topics"]:
            predictions.append(self.__predict_single_news(cand_topic, impression_data["history_topics"]))
        
        return predictions

    def score_news_group(self, impression_group):
        cand_news_topics_group : list = impression_group["candidates_topics"] 

        news_scores = numpy.array([])
        
        self.__logger.debug(f"Candidate news topics: {cand_news_topics_group}")
        for cand_topics in cand_news_topics_group:
            news_score = self.__get_candidate_news_score(cand_topics, impression_group["history_topics"])
            news_scores = numpy.append(news_scores, news_score)

        self.__logger.debug(f"Candidate news scores: {news_scores}")

        return news_scores
    
    def order_news(self, impression_data):        
        cand_news = impression_data["candidate_news_index"]
        self.__logger.debug(f"Candidate news indexes: {cand_news}")
        
        news_scores = self.score_news_group(impression_data)        
        ordered_news = sorted(cand_news, key=lambda x: news_scores[cand_news.index(x)], reverse=True)
        self.__logger.debug(f"Sorted Candidate news indexes: {ordered_news}")

        return ordered_news
    