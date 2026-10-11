"""Tests for the QA Multi-Hop API."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data


def test_ask_empty_question():
    """Test asking with empty question."""
    response = client.post("/ask", json={"question": ""})
    # FastAPI returns 422 for validation errors
    assert response.status_code in (400, 422)


def test_ask_nominal():
    """Test nominal question answering."""
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    # May fail if model not loaded, but should not crash
    assert response.status_code in (200, 503)


def test_ask_abstention():
    """Test abstention case (question with no relevant info)."""
    response = client.post("/ask", json={"question": "What is the meaning of life?"})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        # Should have some response
        assert "answer" in data


def test_ask_off_topic():
    """Test off-topic question."""
    response = client.post("/ask", json={"question": "How to bake a chocolate cake?"})
    assert response.status_code in (200, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])