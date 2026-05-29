"""
MedAI Nexus — Backend Test Suite
PyTest tests for: auth, chat, reports, emergency, agents
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from main import app
from agents.ocr_agent import OCRAgent
from agents.emergency_agent import EmergencyDetectionAgent


# ─── Fixtures ─────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.fixture
def mock_user():
    return {
        "id": "test-user-123",
        "email": "test@medai.dev",
        "full_name": "Test User",
        "is_active": True,
    }


@pytest.fixture
def auth_headers(mock_user):
    from core.auth import create_access_token
    token = create_access_token({"sub": mock_user["id"]})
    return {"Authorization": f"Bearer {token}"}


# ─── Health Check ──────────────────────────────
class TestHealth:
    @pytest.mark.asyncio
    async def test_root(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "online"
        assert r.json()["platform"] == "MedAI Nexus"

    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "services" in data


# ─── Auth Endpoints ───────────────────────────
class TestAuth:
    @pytest.mark.asyncio
    async def test_register_success(self, client):
        r = await client.post("/api/v1/auth/register", json={
            "email": "newuser@medai.dev",
            "password": "SecurePass123!",
            "full_name": "New User",
        })
        assert r.status_code == 201
        data = r.json()
        assert "access_token" in data
        assert data["user"]["email"] == "newuser@medai.dev"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client):
        payload = {"email": "dup@medai.dev", "password": "Pass123!", "full_name": "Dup"}
        await client.post("/api/v1/auth/register", json=payload)
        r = await client.post("/api/v1/auth/register", json=payload)
        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_login_success(self, client):
        # Register first
        await client.post("/api/v1/auth/register", json={
            "email": "login@medai.dev", "password": "Pass123!", "full_name": "Login User",
        })
        r = await client.post("/api/v1/auth/login", json={
            "email": "login@medai.dev", "password": "Pass123!",
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        r = await client.post("/api/v1/auth/login", json={
            "email": "test@medai.dev", "password": "WrongPassword!",
        })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_route_no_token(self, client):
        r = await client.get("/api/v1/auth/me")
        assert r.status_code == 401


# ─── OCR Agent ────────────────────────────────
class TestOCRAgent:
    def test_clean_text(self):
        agent = OCRAgent()
        messy = "Hello   World\n\n\nTest   123"
        clean = agent._clean_text(messy)
        assert "   " not in clean

    def test_parse_hemoglobin(self):
        agent = OCRAgent()
        text = "Hemoglobin: 11.2 g/dL"
        values = agent._parse_medical_values(text)
        assert "hemoglobin" in values
        assert values["hemoglobin"] == "11.2"

    def test_parse_blood_pressure(self):
        agent = OCRAgent()
        text = "BP: 140/90 mmHg"
        values = agent._parse_medical_values(text)
        assert "blood_pressure" in values

    def test_parse_blood_sugar(self):
        agent = OCRAgent()
        text = "Blood sugar: 126 mg/dL (FBS)"
        values = agent._parse_medical_values(text)
        assert "blood_sugar" in values

    @pytest.mark.asyncio
    async def test_missing_file_raises(self):
        agent = OCRAgent()
        with pytest.raises(FileNotFoundError):
            await agent.extract("/nonexistent/path/report.pdf")


# ─── Emergency Agent ─────────────────────────
class TestEmergencyAgent:
    def test_fast_triage_chest_pain(self):
        agent = EmergencyDetectionAgent(None)
        assert agent.fast_triage("I have severe chest pain") is True

    def test_fast_triage_stroke(self):
        agent = EmergencyDetectionAgent(None)
        assert agent.fast_triage("Face drooping and arm weakness") is True

    def test_fast_triage_normal(self):
        agent = EmergencyDetectionAgent(None)
        assert agent.fast_triage("I have a mild headache") is False

    def test_urgent_detection(self):
        agent = EmergencyDetectionAgent(None)
        assert agent.is_urgent("High fever and severe headache") is True

    @pytest.mark.asyncio
    async def test_fast_path_returns_critical(self):
        agent = EmergencyDetectionAgent(None)
        result = await agent.assess("I am having a heart attack, chest pain is unbearable")
        assert result["is_emergency"] is True
        assert result["severity"] == "critical"
        assert result["call_ambulance"] is True


# ─── Chat Endpoint ────────────────────────────
class TestChat:
    @pytest.mark.asyncio
    @patch("routes.chat.orchestrator.run_chat", new_callable=AsyncMock)
    async def test_send_message(self, mock_chat, client, auth_headers):
        mock_chat.return_value = {
            "response": "Based on your symptoms, you may have...",
            "is_emergency": False,
            "severity": "low",
            "sources": ["medical_reference"],
        }
        r = await client.post("/api/v1/chat/message",
            json={"content": "What are symptoms of diabetes?"},
            headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "assistant"
        assert "response" in data["content"] or data["content"]

    @pytest.mark.asyncio
    @patch("routes.chat.orchestrator.run_chat", new_callable=AsyncMock)
    async def test_emergency_in_chat(self, mock_chat, client, auth_headers):
        mock_chat.return_value = {
            "response": "🚨 CRITICAL — Call 112 immediately.",
            "is_emergency": True,
            "severity": "critical",
            "sources": [],
        }
        r = await client.post("/api/v1/chat/message",
            json={"content": "I have severe chest pain and can't breathe"},
            headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["is_emergency"] is True


# ─── Reports Endpoint ─────────────────────────
class TestReports:
    @pytest.mark.asyncio
    async def test_upload_no_auth(self, client):
        r = await client.post("/api/v1/reports/upload",
            files={"file": ("test.pdf", b"test content", "application/pdf")})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_list_reports(self, client, auth_headers):
        r = await client.get("/api/v1/reports", headers=auth_headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ─── Rate Limiting ────────────────────────────
class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, client):
        # Hammer the health endpoint
        responses = []
        for _ in range(70):
            r = await client.get("/api/health")
            responses.append(r.status_code)
        # Some requests should be rate limited
        assert 429 in responses
