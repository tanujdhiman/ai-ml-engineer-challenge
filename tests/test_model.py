import pytest
from test import MBTIPredictor

@pytest.fixture
def predictor():
    return MBTIPredictor()

def test_predict(predictor):
    # Test with valid input
    text = "I love deep conversations and thinking about abstract ideas."
    mbti_type, confidence = predictor.predict(text)
    
    assert isinstance(mbti_type, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_predict_empty_input(predictor):
    # Test with empty input
    with pytest.raises(ValueError):
        predictor.predict("")

def test_predict_batch(predictor):
    # Test batch prediction
    texts = [
        "I enjoy socializing with large groups of people.",
        "I prefer spending time alone reading books."
    ]
    results = predictor.predict_batch(texts)
    
    assert len(results) == 2
    for result in results:
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)
