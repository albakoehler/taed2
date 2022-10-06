import yaml
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import re


# Path of the parameters file
params_path = "params.yaml"

# Path of the input data folder
input_folder_path = Path("data/raw")

# Path of the files to read
data_path = input_folder_path / "raw_data.csv"

# Read dataset from csv file
df = pd.read_csv(data_path, encoding = 'latin', header=None)

# Read data preparation parameters

with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

### Data set processing

# rename columns for reference
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

# drop data fields we don't need
df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)

# change numeric labels to text
lab_to_sentiment = {0:"Negative", 4:"Positive"}
def label_decoder(label):
    return lab_to_sentiment[label]
df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))

### Text processing

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

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

df.text = df.text.apply(lambda x: preprocess(x))

# Split into train and test sets
MAX_NB_WORDS = 100000

train_data, test_data = train_test_split(df, test_size=1-params['train_size'],random_state=params['random_state'])

# Separate target from predictors
y_train = train_data.sentiment
y_test = test_data.sentiment

X_train = train_data.drop(["sentiment"], axis=1)
X_test = test_data.drop(["sentiment"], axis=1)


# Path of the output data folder
Path("data/processed")
prepared_folder_path = Path("data/processed")

X_train_path = prepared_folder_path / "X_train.csv"
y_train_path = prepared_folder_path / "y_train.csv"
X_test_path = prepared_folder_path / "X_test.csv"
y_test_path = prepared_folder_path / "y_test.csv"

X_train.to_csv(X_train_path, index=False)
print("Writing file {} to disk.".format(X_train_path))

y_train.to_csv(y_train_path, index=False)
print("Writing file {} to disk.".format(y_train_path))

X_test.to_csv(X_test_path, index=False)
print("Writing file {} to disk.".format(X_test_path))

y_test.to_csv(y_test_path, index=False)
print("Writing file {} to disk.".format(y_test_path))
