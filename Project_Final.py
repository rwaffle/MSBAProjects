import pandas as pd
import nltk
import twint
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import string, re
import sys

sys.setrecursionlimit(1500)
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'http\S+'
pat3 = r'pic.twitter.com\S+'
combined_pat = r'|'.join((pat1, pat2, pat3))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    lettersAndNumbers = re.sub("[^A-Za-z0-9]", " ", clean)
    lowerCase = lettersAndNumbers.lower()
    words = tok.tokenize(lowerCase)
    return (" ".join(words)).strip()

sid = SentimentIntensityAnalyzer()

movieDictionary = {}
data = pd.read_csv('MoviesKeyword.csv')
movieDictionary = {col: list(data[col]) for col in data.columns}
for movie in movieDictionary:
    pos_cnt = 0
    neu_cnt = 0
    neg_cnt = 0
    dateList = []
    phase1List = []
    phase2List = []
    phase3List = []
    sentimentList = []
    tweetList = []
    aggList = ['mean_sentiment_polarity', 'count_tweets', 'sum_likes', 'sum_retweets']
    tweetFile = movie + '.csv'
    c = twint.Config()
    c.Hide_output = True
    c.Lang = 'en'
    c.Format = "{id}|{date}|{time}|{tweet}|{nlikes}|{nretweets}|{hashtags}"
    c.Search = movieDictionary[movie][0]
    c.Since = movieDictionary[movie][1]
    c.Until = movieDictionary[movie][6]
    c.Pandas = True
    c.Store_pandas = True
    twint.run.Search(c)
    movieDF = twint.storage.panda.Tweets_df
    dateList.append(movieDictionary[movie][1])
    dateList.append(movieDictionary[movie][2])
    dateList.append(movieDictionary[movie][3])
    dateList.append(movieDictionary[movie][4])
    dateList.append(movieDictionary[movie][5])
    dateList.append(movieDictionary[movie][6])
    movieDF['date'] = pd.to_datetime(movieDF['date'], format = '%Y-%m-%d')
    movieDF = movieDF.drop(['id', 'conversation_id', 'created_at', 'timezone','place', 'cashtags', 'user_id', 'user_id_str', 'username', 'name', 'day', 'hour', 'link', 'retweet', 'nreplies', 'quote_url', 'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date'], axis = 1)
    tweets = movieDF['tweet'].values.tolist()
    for t in tweets:
        cleanedTweet = tweet_cleaner(t)
        tweetList.append(cleanedTweet)
        sentimentScore = sid.polarity_scores(cleanedTweet)
        sentimentList.append(sentimentScore['compound'])
        if sentimentScore['compound'] > 0:
            pos_cnt += 1
        elif sentimentScore['compound'] == 0:
            neu_cnt += 1
        elif sentimentScore['compound'] < 0:
            neg_cnt += 1
    cnt = {'Movie' : movie, 'positive_count': pos_cnt, 'neutral_count': neu_cnt, 'negative_count': neg_cnt}
    countHeader = ['Movie', 'positive_count', 'neutral_count', 'negative_count']
    countDF = pd.DataFrame(cnt, index = [0])
    countDF.columns = countHeader
    countDF.to_csv('SentimentCountFile.csv', mode = 'a', header = True)
    movieDF['cleaned_tweet'] = tweetList
    movieDF['sentiment_polarity'] = sentimentList  
    movieDF = movieDF.sort_values(by=['date'], ascending = True)
    movieDF = movieDF.set_index(['date'])
    movieByDay = movieDF.resample('D').agg({'sentiment_polarity':'mean','tweet':'count','nlikes':'sum','nretweets':'sum'})
    phase1Aggs = movieDF.loc[dateList[0]:dateList[1]].agg({'sentiment_polarity':'mean','tweet':'count','nlikes':'sum','nretweets':'sum'})
    phase2Aggs = movieDF.loc[dateList[2]:dateList[3]].agg({'sentiment_polarity':'mean','tweet':'count','nlikes':'sum','nretweets':'sum'})
    phase3Aggs = movieDF.loc[dateList[4]:dateList[5]].agg({'sentiment_polarity':'mean','tweet':'count','nlikes':'sum','nretweets':'sum'})
    phase1AggsList = phase1Aggs.values.tolist()
    phase2AggsList = phase2Aggs.values.tolist()
    phase3AggsList = phase3Aggs.values.tolist()
    d = {'Movie': movie,'Data': aggList, 'Phase1':phase1AggsList, 'Phase2':phase2AggsList, 'Phase3':phase3AggsList}
    movieByPhase = pd.DataFrame(d).set_index(['Movie'])
    dailyFile = movie + ' daily.csv'
    movieDF.to_csv(tweetFile,sep ='|', header = True)
    movieByDay.to_csv(dailyFile, header = True)
    movieByPhase = movieByPhase.transpose()
    movieByPhase.to_csv('FinalDataFile.csv', mode = 'a', header = True)
    print(movie + ' complete')




