"""Tests de integración para los endpoints de la API."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_model_info(client):
    response = await client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "not_trained")


@pytest.mark.asyncio
async def test_recommendations_native_format(client):
    response = await client.post("/api/v1/recommendations", json={
        "doctor": {
            "id": "doc-test",
            "name": "Dr. Test",
            "specialty": "Cardiología",
        },
        "top_k": 5,
    })
    assert response.status_code in (200, 503)  # 503 if model not trained


@pytest.mark.asyncio
async def test_recommendations_frontend_format(client):
    response = await client.post("/api/v1/recommendations", json={
        "doctor_id": "uuid-test-123",
        "doctor_profile": {
            "specialty": "Cardiología",
            "hospital": "Hospital Nacional",
            "years_experience": 10,
        },
        "top_k": 5,
    })
    assert response.status_code in (200, 503)


@pytest.mark.asyncio
async def test_risk_assessment(client):
    response = await client.post("/api/v1/risk-assessment", json={
        "specialty": "Cardiología",
        "priority": "alta",
        "procedure_complexity": "alta",
        "documentation_complete": True,
        "informed_consent": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "recommendations" in data
    assert 0.0 <= data["risk_score"] <= 1.0


@pytest.mark.asyncio
async def test_risk_assessment_missing_docs(client):
    response = await client.post("/api/v1/risk-assessment", json={
        "specialty": "Neurocirugía",
        "priority": "critica",
        "documentation_complete": False,
        "informed_consent": False,
        "has_prior_complaints": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ("alto", "critico")
