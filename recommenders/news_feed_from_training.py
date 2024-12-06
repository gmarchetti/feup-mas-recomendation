def build_news_feed(user_session: list):
    full_news_feed =[]
    positive_news = user_session[2]
    negative_news = user_session[3]

    full_news_feed = positive_news + negative_news
    
    return full_news_feed