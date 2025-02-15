import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_predict_endpoint(client):
    # Test valid input
    response = client.post(
        "/predict",
        json={"text": "I love deep conversations and thinking about abstract ideas."}
    )
    
    assert response.status_code == 200
    assert "mbti_type" in response.json()
    assert "confidence" in response.json()
    assert isinstance(response.json()["confidence"], float)

def test_predict_invalid_input(client):
    # Test invalid input (non-string)
    response = client.post(
        "/predict",
        json={"text": 123}
    )
    
    assert response.status_code == 422  # FastAPI validation error
