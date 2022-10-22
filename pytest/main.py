import pickle
from datetime import datetime
import pandas as pd
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, Request

# Load model
model_folder_path = Path("models")
with open(model_folder_path / "lstm_model.pkl", "rb") as pickled_model:
    model = pickle.load(pickled_model)

# create tokenizer for the input text that we are going to predict
x_train = pd.read_csv("src/data/processed/X_train.csv")
x_train.text= x_train.text.astype(str)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train.text)
word_index = tokenizer.word_index
MAX_SEQUENCE_LENGTH = 30

# Define application
app = FastAPI(
    title="Moody Tweets",
    description="This API lets you make predictions bout the sentiment of a short text.",
    version="0.1",
)

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""
    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to moody tweets classifier!"},
    }
    return response

 # Auxiliar function to get the prediction type
def decode_sentiment(score):
    return "Positive" if score>0.5 else "Negative"

@app.post("/prediction/", tags=["Prediction"])
@construct_response
def sentiment_prediction(request: Request, text: str = ""):
    if text == "":
        # non valid input
        response = {"message": "Empty string", "status-code": HTTPStatus.BAD_REQUEST}
    else:
        # get prediction
        inputtext = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen = MAX_SEQUENCE_LENGTH)
        score = model.predict(inputtext)
        prediction = decode_sentiment(score[0])
        x = score[0]
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {"prediction": prediction, "score" : float(x[0])},
        }
    return response
