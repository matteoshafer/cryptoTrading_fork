import os
import re
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import torch
from audioread.ffdec import ReadTimeoutError
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import CONSTANTS
from CONSTANTS import *


def preprocess(str):
    # Remove punctuation, whitespace, and special characters, standardize case
    pattern = r'[^\w\s]|[\n\r\t]'
    preprocessed = re.sub(pattern, '', str).upper()
    return preprocessed

def get_newspapers(query):
  params = {
    "api_key": "d8b42ca90077ec59802d53ad3d55e7e05d38ebac54426fbfa8913b8f40f68e24",
    "engine": "google_news",
    "hl": "en",
    "gl": "us",
    "q": query
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  article_metadata = results['news_results']
  newspapers = []
  MISSING = '-'

  for article in tqdm(article_metadata):
      article_link = article.get('link', MISSING)
      article_title = article.get('title', MISSING)
      article_publication_date = article.get('date', MISSING)
      try:
          time.sleep(5)
          response = requests.get(article_link, timeout=30)
          response.raise_for_status()
          soup = BeautifulSoup(response.content, 'html.parser')
          article_text = ""
          for paragraph in soup.find_all('p'):
              article_text += paragraph.get_text() + "\n"

          items = {
              'title': article_title,
              'date': article_publication_date,
              'link': article_link,
              'text': article_text,
          }
          newspapers.append(items)
      except requests.exceptions.RequestException as e:
          pass
      except Exception as e:
          pass
  return pd.DataFrame(newspapers)

def sentimentAnalysis(newspapers_df, NEGATIVE, NEUTRAL, POSITIVE):
    """STEP 2"""
    newspapers_df['things'] = newspapers_df['text'].astype(str).apply(get_sentiment)
    newspapers_df['sentiment'] = newspapers_df['things'].apply(lambda x: x[0])
    newspapers_df['score'] = newspapers_df['things'].apply(lambda x: 100 * (NEGATIVE * x[1][0] + NEUTRAL * x[1][1] + POSITIVE * x[1][2]))
    newspapers_df.drop('things', axis=1, inplace=True)
    return newspapers_df

def get_sentiment(text):
    """HELPER FUNCTION"""
    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokens = tokenizer(preprocess(text), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    probabilities = softmax(outputs.logits, dim=-1) # Activation function to get predicted class probabilities
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    label_mapping = model.config.id2label
    predicted_label = label_mapping[predicted_class] # map it back to a readable label
    return predicted_label, probabilities.tolist()[0] # return predicted label and associated probabilities


def newspapers_from_queries(coin, queries_path):
    """STEP 1"""
    queries = None
    with open(queries_path) as queries:
        query = queries.read().splitlines()
        query = [f'{coin} {q}' for q in query]
        queries = query

    PATH = f'{coin}_newspapers.csv'
    newspapers = pd.DataFrame()
    if os.path.exists(PATH):
        newspapers = pd.read_csv(PATH)
    for q in tqdm(queries):
        stuff = get_newspapers(q)
        newspapers = pd.concat([newspapers, stuff], ignore_index=True)

    newspapers.to_csv(PATH)
    return newspapers

def newspaper_sentiment_pipeline(coin, queries_path='queries.txt', NEGATIVE=-1, NEUTRAL=0, POSITIVE=1):
    # Step 1: get newspapers from queries
    nfq = newspapers_from_queries(coin, queries_path)
    
    # Step 2: sentiment analysis
    nfs = sentimentAnalysis(nfq, NEGATIVE, NEUTRAL, POSITIVE)
    
    # Step 3: Load in the newspaper data (with sentiment) and preprocess it
    coin_newspapers = pd.read_csv(f'{coin}_newspapers.csv')
    coin_newspapers['date'] = pd.to_datetime(coin_newspapers['date'], format="%m/%d/%Y, %I:%M %p, %z UTC")
    coin_newspapers['date'] = coin_newspapers['date'].dt.date
    
    # Step 4: Merge the newspaper data with the full/market data
    df = pd.read_csv(fullDataPath(coin))
    df['time'] = pd.to_datetime(df['time'])  # Convert 'time' to datetime
    coin_newspapers['date'] = pd.to_datetime(coin_newspapers['date'])  # Convert 'date' to datetime
    merged_df = pd.merge(df, coin_newspapers, left_on='time', right_on='date', how='inner')
    merged_df.to_csv(fullDataPath(coin), index=False)

def fullDataPath(coin):
    return f'fulldata/{coin}_df.csv'

def get_fgi_data():
  url = f'https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical?CMC_PRO_API_KEY={CMC_KEY}&limit={LIMIT}'
  json = requests.get(url)
  if json.status_code == 200:
      data = json.json()
      fgi_df = pd.DataFrame(data['data'])
      fgi_df['timestamp'] = pd.to_datetime(fgi_df['timestamp'], unit='s')
      return fgi_df
  else:
    print(f'Couldn\'t do it because response code is {json.status_code}')
    return None

def myFillNa(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(0, inplace=True)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col].fillna(CONSTANTS.EMPTY_STRING, inplace=True)
        else:
            df[col].fillna(np.nan, inplace=True)

def cv_metrics(model, data, yCol='gradient', v=5, trainingColsPath='training_columns.txt'):
    model = DecisionTreeRegressor(max_depth=5)
    trainingCols = open('training_columns.txt', 'r').readlines()
    trainingCols = [i.strip() for i in trainingCols]
    assert yCol not in trainingCols, f'{yCol} should not be in trainingCols but was found in it'
    myFillNa(data)
    X = pd.get_dummies(data[trainingCols])
    y = data[yCol]
    cv_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_root_mean_squared_error')
    cv_scores = pd.Series(cv_scores)
    cv_scores.index += 1
    cv_scores.plot.bar()
    print(f'CV RMSE: {cv_scores.mean()}')
    return cv_scores

def setup(coin, targetCol='gradient', closeCol='close'):
    data = pd.read_csv(fullDataPath(coin))
    trainingCols = open(TRAINING_COLUMNS, 'r').readlines()
    trainingCols = [i.strip() for i in trainingCols]
    setDiff = np.setdiff1d(trainingCols, data.columns)
    assert np.isin(trainingCols, data.columns).all(), f'{", ".join(setDiff)} not in data'
    X = data[trainingCols]
    data[targetCol] = data[closeCol].diff().fillna(0.0)
    data['TextType'] = data['link'].apply(lambda x: 'tweet' if x == CONSTANTS.EMPTY_STRING else 'newspaper')
    y = data[targetCol]
    return data, X, y

def prices(product_id, period=30, granularity=86400, start=None, end=None):
    """
    Fetch historical candlestick data for a cryptocurrency pair from now to the specified number of days in the past.

    :param product_id: The product ID for the crypto pair (e.g., 'BTC-USD').
    :param period: Number of days of historical data to fetch.
    :param granularity: Desired time slice in seconds (60, 300, 900, 3600, 21600, 86400).
    :return: DataFrame containing historical data.
    """
    if not product_id.endswith('-USD'):
        product_id += '-USD'
    product_id = product_id.upper()
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    if start is None and end is None: # get data from specified number of days ago if date bounds are not specified.
        end = datetime.now()
        start = end - timedelta(days=period)
    coin = product_id.split('-')[0]
    all_data = []

    while start < end:
        end_slice = min(start + timedelta(seconds=granularity * 300), end)
        params = {
            'start': start.isoformat(),
            'end': end_slice.isoformat(),
            'granularity': granularity
        }

        try:
            response = requests.get(url, params=params)
        except ConnectionError:
            print("No internet connection")
            return None, coin
        except ReadTimeoutError:
            print('Your wifi likely doesn\'t allow to access Coinbase API')
            return None, coin

        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
        else:
            print("Failed to fetch data:", response.text)
            break

        start = end_slice

    if all_data:
        columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        data = pd.DataFrame(all_data, columns=columns)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data['change'] = data['close'] - data['open']
        data['pct_change'] = (data['change'] / data['open']) * 100
        return data, coin
    return None, coin