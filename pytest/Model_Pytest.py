import pytest

import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#TEST 1: LABELING
#check if we get the correct label
def decode_sentiment(score):
    return "Positive" if score>0.5 else "Negative"
    
#We have 3 experiments: 
#1) As >0.5 is positive we check it with 0.8
#2) As <0.5 is positive we check it with 0.1
#3) We check whether having an incorrect label is seeing as a FAIL experiment
@pytest.mark.parametrize("score,expected_decode", [
    (0.8, "Positive"),
    (0.1,"Negative"),
    (0.7, "Negative"),
])

def test_decode_sentiment(score, expected_decode):
  assert decode_sentiment(score) == expected_decode




#TEST 2: STEMMING
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = r'\n@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+\n'


def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

#We check an example of stemming traveler -> travel
@pytest.mark.preproces
def test_preprocess():
  my_pre = preprocess("traveler",True)
  expected_pre = 'travel'
  assert my_pre == expected_pre
