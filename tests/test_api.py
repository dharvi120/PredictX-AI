from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["data"]["maintenance_records"] == 100


def test_dashboard_endpoint():
    response = client.get("/api/dashboard")
    assert response.status_code == 200

    data = response.json()

    assert data["summary"]["total_engines"] == 100
    assert len(data["engines"]) == 100

    assert data["summary"]["high_priority"] == 19
    assert data["summary"]["medium_priority"] == 81
    assert data["summary"]["low_priority"] == 0


def test_individual_engine_endpoint():
    response = client.get("/api/engines/1")
    assert response.status_code == 200

    data = response.json()

    assert data["engine"]["unit_id"] == 1
    assert len(data["sensors"]) == 21


def test_missing_engine_returns_404():
    response = client.get("/api/engines/9999")
    assert response.status_code == 404


def test_explainability_endpoint():
    response = client.get("/api/explainability")
    assert response.status_code == 200

    data = response.json()

    assert len(data["global_importance"]) == 15
    assert len(data["local_explanation"]) == 15

    assert data["global_importance"][0]["feature"] == "cycle"
