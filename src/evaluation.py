import pickle
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

# Path to the prepared data folder
input_folder_path = Path("data/processed")

# Path to the models folder
model_folder_path = Path("../models")

# Path to the metrics folder
metrics_folder_path = Path("../metrics")

# Read validation dataset
x_test = pd.read_csv(input_folder_path / "X_test.csv")
x_train = pd.read_csv(input_folder_path / "X_train.csv")
y_test = pd.read_csv(input_folder_path / "y_test.csv")

# pad sequences and convert to string to avoid errors

x_test.text= x_test.text.astype(str)
x_train.text= x_train.text.astype(str)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train.text)
word_index = tokenizer.word_index

x_test.text= x_test.text.astype(str)
MAX_SEQUENCE_LENGTH = 30
x_test = pad_sequences(tokenizer.texts_to_sequences(x_test.text), maxlen = MAX_SEQUENCE_LENGTH)


def decode_sentiment(score):
    """give the sentiment evaluation based on the score"""
    return "Positive" if score>0.5 else "Negative"

# Load the model
with open(model_folder_path / "lstm_model.pkl", "rb") as pickled_model:
    model = pickle.load(pickled_model)

# make predictions with test set
scores = model.predict(x_test, verbose=1, batch_size=10000)
y_pred_1d = [decode_sentiment(score) for score in scores]

acc = accuracy_score(list(y_test.sentiment), y_pred_1d)
neg_precision = precision_score(list(y_test.sentiment), y_pred_1d, pos_label='Negative')
pos_precision = precision_score(list(y_test.sentiment), y_pred_1d, pos_label='Positive')
pos_recall = recall_score(list(y_test.sentiment), y_pred_1d, pos_label='Positive')
neg_recall = recall_score(list(y_test.sentiment), y_pred_1d, pos_label='Negative')

# Write scores
with open(metrics_folder_path / "scores.json", "w") as scores_file:
    json.dump(
        {"Accuracy": acc, "Negative Precision": neg_precision,
         "Positive Precision" : pos_precision, "Negative Recall" : neg_recall,
        "Positive Recall" : pos_recall}, scores_file, indent=4,
    )

print("Evaluation completed.")
