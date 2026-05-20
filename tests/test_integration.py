"""
Integration tests — run against real services (Qdrant, Redis, LLMs).
Only run in CI on main/staging branches, not on every PR.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)

@pytest.mark.integration
class TestEndtoEndChat:
    """Test that require qdrant + LLM apis"""

    def test_factual_question_returns_answer(self):
        r = client.post("/api/v1/chat", json={
            "question": "How many vacation days do full-time employees get?",
            "thread_id": "integration_test_1",
        })

        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 20
        # Should mention vacation days
        assert "vacation" in data["answer"].lower() or "days" in data["answer"]

    def test_chat_question_no_sources(self):
        r = client.post("/api/v1/chat", json={
            "question": "Hello!",
            "thread_id": "integration_test_2",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("sources", [])) == 0

    def test_out_of_scope_refused(self):
        r = client.post("/api/v1/chat", json={
            "question": "What is the weather in San Francisco?",
            "thread_id": "integration_test_3",
        })
        assert r.status_code == 200
        data = r.json()
        answer_lower = data["answer"].lower()
        assert any(phrase in answer_lower for phrase in [
            "hr", "policy", "can only help", "cannot",
        ])

    def test_cache_works(self):
        question = "What is the overtime pay rate?"
        r1 = client.post("/api/v1/chat", json={
            "question": question, "thread_id": "cache_test",
        })
        assert r1.status_code == 200

        import time
        time.sleep(2)  # Let cache warm up

        r2 = client.post("/api/v1/chat", json={
            "question": question, "thread_id": "cache_test_2",
        })
        assert r2.status_code == 200
        assert r1.json()["answer"] == r2.json()["answer"]


@pytest.mark.integration
class TestMultiHopQuery:
    """Tests that specifically validate query decomposition."""

    def test_comparision_question_decompose(self):
        r = client.post("/api/v1/chat", json={
            "question": "How does FMLA differ from personal leave?",
            "thread_id": "multihop_test",
        })

        assert r.status_code == 200
        answer = r.json()["answer"]
        # should mention both policies
        assert len(answer) > 100 # comparision answers are longer