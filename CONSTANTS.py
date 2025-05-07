EMPTY_STRING = '-'
CMC_KEY = "98464488-2db9-4ddc-9b98-6b48f8b623dc"
LIMIT = 365
TRAINING_COLUMNS = 'training_columns.txt'
COIN = 'BTC'
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'newspapers/{COIN}_newspapers.csv'

# Early Stopping Constants
LOSS_MULTIPLIER = 1
LOSS_10E_THING = -8
LOSS_THRESHOLD = LOSS_MULTIPLIER * 10^LOSS_10E_THING