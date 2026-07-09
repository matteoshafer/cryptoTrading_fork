import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMPTY_STRING = '-'
CMC_KEY = os.environ.get("CMC_KEY", "")
LIMIT = 365
TRAINING_COLUMNS = 'training_columns.txt'
COIN = 'ETH'
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'newspapers/{COIN}_newspapers.csv'
SLEEP = 3
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")

# Early Stopping Constants
LOSS_MULTIPLIER = 1
LOSS_10E_THING = -8
LOSS_THRESHOLD = LOSS_MULTIPLIER * 10 ** LOSS_10E_THING