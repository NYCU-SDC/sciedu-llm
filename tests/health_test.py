import os
os.environ["OPENAI_API_KEY"] = "mock_key"
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
