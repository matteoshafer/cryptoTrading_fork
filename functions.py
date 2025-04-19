import re
import os
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
    newspapers_df['score'] = newspapers_df['things'].apply(lambda x: 100 * (NEGATIVE * x[1][0] + NEUTRAL * x[0][1] + POSITIVE * x[1][2]))
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
    df = pd.read_csv(f'fulldata/{coin}_df.csv')
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
    print("Couldn\'t get FGI data with status code" + str(json.status_code))
    return None