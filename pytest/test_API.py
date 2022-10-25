import pickle
from datetime import datetime
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.json() == {"message": "Welcome to moody tweets classifier!"}


def test_decode_sentiment():
	my_decode = decode_sentiment(5)
	expected_decode = "Positive"
	
	assert my_decode == expected_decode

def test_create_item():
	response = client.post("/prediction/", json={""})
	assert response.json() == {"message": "Empty string", "status-code": HTTPStatus.BAD_REQUEST}
