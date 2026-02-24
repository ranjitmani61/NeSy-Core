"""
tests/integration/test_server.py
==================================
Integration tests for the FastAPI inference server.
Tests the REST API contract: /reason, /learn, /health endpoints.

Requires: pip install httpx (for async test client)
Falls back to direct route-function testing if httpx unavailable.
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from nesy.deployment.server.app import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_health_includes_framework(self, client):
        resp = client.get("/health")
        assert resp.json()["framework"] == "nesy-core"


class TestReasonEndpoint:
    def test_reason_with_empty_facts(self, client):
        resp = client.post(
            "/api/v1/reason",
            json={
                "facts": [],
                "context_type": "general",
                "neural_confidence": 0.90,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "status" in body
        assert "confidence" in body
        assert set(body["confidence"].keys()) == {"factual", "reasoning", "knowledge_boundary"}

    def test_reason_with_valid_facts(self, client):
        resp = client.post(
            "/api/v1/reason",
            json={
                "facts": [
                    {"name": "HasSymptom", "args": ["patient_1", "fever"]},
                ],
                "context_type": "medical",
                "neural_confidence": 0.85,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "flagged", "uncertain", "rejected")
        assert isinstance(body["reasoning_steps"], int)

    def test_reason_returns_trustworthy_flag(self, client):
        resp = client.post(
            "/api/v1/reason",
            json={
                "facts": [],
                "neural_confidence": 0.95,
            },
        )
        body = resp.json()
        assert isinstance(body["trustworthy"], bool)

    def test_reason_bad_request_missing_facts(self, client):
        resp = client.post("/api/v1/reason", json={})
        assert resp.status_code == 422  # Pydantic validation error


class TestLearnEndpoint:
    def test_learn_adds_rule(self, client):
        resp = client.post(
            "/api/v1/learn",
            json={
                "rule_id": "test_rule_server",
                "antecedents": [["HasSymptom", "?p", "headache"]],
                "consequents": [["MayHave", "?p", "migraine"]],
                "weight": 0.80,
                "make_anchor": False,
                "description": "Headache → migraine (test)",
            },
        )
        assert resp.status_code == 200

    def test_learn_with_anchor(self, client):
        resp = client.post(
            "/api/v1/learn",
            json={
                "rule_id": "test_anchor_server",
                "antecedents": [["A", "?x"]],
                "consequents": [["B", "?x"]],
                "weight": 1.0,
                "make_anchor": True,
            },
        )
        assert resp.status_code == 200


class TestRulesEndpoint:
    def test_rules_list(self, client):
        resp = client.get("/api/v1/rules")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, (list, dict))
