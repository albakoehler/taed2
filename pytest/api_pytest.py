# -*- coding: utf-8 -*-
"""API_pytest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EX3-WS1ufwTqOYn8Y6LYoBGJe_rcTYuV
"""

from fastapi.testclient import TestClient

from .api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.json() == {"message": "Welcome to moody tweets classifier!"}


def test_decode_sentiment():
  my_decode = decode_sentiment(5)
  expected_decode = "Positive"

  assert my_decode == expected_decode

 response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {"prediction": prediction, "score" : float(x[0])},

def test_create_item():
    response = client.post(
        "/prediction/",
        json={"message": "foobar", "status-code": 548451, "data": {"prediction":"The Foo Barters",  "score" :200}},
    )
    assert response.json() == {
        "message": "foobar",
        "status-code": 548451,
        "data": {"prediction": "The Foo Barters", "score" : float(x[0])},
    }