import json
from nltk.tokenize import word_tokenize

with open('Geo files/geography.json') as f:
    geo_data = json.load(f)