import pandas as pd
import re
from textblob import TextBlob # pip install textblob

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # remove links
    text = re.sub(r'[^\w\s]', '', text)
    return text

def analyze_intent_score(text):
    # Dictionnaire étendu
    intent_words = ["visit", "travel", "going", "plan", "booking", "flight", "hotel", 
                    "bucket list", "next trip", "destination", "someday", "stunning"]
    score = sum(1 for word in intent_words if word in text.lower())
    return 1 if score > 0 else 0

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity # -1 (négatif) à 1 (positif)