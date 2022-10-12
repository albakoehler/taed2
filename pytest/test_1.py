
import pytest

def decode_sentiment(score):
    return "Positive" if score>0.5 else "Negative"


def test_decode1():
  my_decode = decode_sentiment(5)
  expected_decode = "Positive"

  assert my_decode == expected_decode

#pytest --> per saber si al canviar el nom de la funció obtenim
#           el valor esperat
#marquem aquest test com decode1 per aixi al executar pytest poder
#escollir executar nomes aquest test

@pytest.mark.decode2
def test_decode2():
  my_decode = decode_sentiment(0.1)
  expected_decode = "Negative"

  assert my_decode == expected_decode





