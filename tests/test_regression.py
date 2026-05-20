"""
Regression tests — structural quality gate using the golden test set.
Runs 5 questions and verifies answers meet minimum quality standards.
Blocks deploy if answers degrade. Fast, free, no RAGAS calls needed.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
GOLDEN_PATH = Path("data/golden_test_set/golden_test_set.json")
REGRESSION_SAMPLE_SIZE = 5

@pytest.fixture
def golden_sample():
    with open(GOLDEN_PATH) as f:
        full_set = json.load(f)
    questions = [q for q in full_set if q.get("category") != "out_of_scope"]
    return questions[:REGRESSION_SAMPLE_SIZE]

@pytest.mark.regression
class TestRegressionSuite:
    def test_answers_are_not_empty(self, golden_sample):
        """Every HR question must return a substantive answer."""
        for item in golden_sample:
            r = client.post("/api/v1/chat", json={
                "question": item["question"],
                "thread_id": f"regression_empty_{item['category']}",
            })
            assert r.status_code == 200, (
                f"API returned {r.status_code} for: {item['question']}"
            )
            assert len(r.json()["answer"]) > 10, (
                f"Empty or too-short answer for: {item['question']}"
            )
    
    def test_no_obvious_hallucinations(self, golden_sample):
        """Answers must not contain LLM hallucination markers."""
        hallucination_phrases = [
            "as an ai language model",
            "i cannot access",
            "based on my training data",
            "i don't have access to",
            "my knowledge cutoff",
        ]
        for item in golden_sample:
            r = client.post("/api/v1/chat", json={
                "question": item["question"],
                "thread_id": f"regression_hallucination_{item['category']}",
            })
            assert r.status_code == 200
            answer_lower = r.json()["answer"].lower()
            for phrase in hallucination_phrases:
                assert phrase not in answer_lower, (
                    f"Hallucination phrase '{phrase}' detected in answer for: "
                    f"{item['question']}"
                )

    def test_factual_questions_mention_key_terms(self, golden_sample):
        """Spot-check that factual answers contain expected terms."""
        key_term_map = {
            "vacation": ["days", "vacation", "15"],
            "sick": ["sick", "days"],
            "overtime": ["overtime", "one-and-a-half", "1.5"],
            "probation": ["90", "probation"],
            "health insurance": ["insurance", "health", "30"],
            "drug": ["drug", "testing", "policy"],
        }
        for item in golden_sample:
            question_lower = item["question"].lower()

            # Skip if no keyword matches this question
            matched_keyword = None
            for keyword in key_term_map:
                if keyword in question_lower:
                    matched_keyword = keyword
                    break

            if matched_keyword is None:
                continue

            r = client.post("/api/v1/chat", json={
                "question": item["question"],
                "thread_id": f"regression_keyterm_{item['category']}",
            })
            assert r.status_code == 200
            answer_lower = r.json()["answer"].lower()

            expected_terms = key_term_map[matched_keyword]
            has_term = any(t in answer_lower for t in expected_terms)
            assert has_term, (
                f"Expected one of {expected_terms} in answer for: "
                f"{item['question']}\nGot: {r.json()['answer'][:200]}"
            )

    def test_out_of_scope_questions_are_refused(self):
        """Out-of-scope questions must be politely refused."""
        out_of_scope = [
            "What is the weather in San Francisco today?",
            "Help me write a Python script for linked list",
            "What is the stock price of Apple?",
        ]
        refusal_indicators = [
            "hr", "policy", "policies", "can only help",
            "cannot help", "not able to", "outside",
        ]
        for question in out_of_scope:
            r = client.post("/api/v1/chat", json={
                "question": question,
                "thread_id": "regression_outofscope",
            })
            assert r.status_code == 200
            answer_lower = r.json()["answer"].lower()
            has_refusal = any(phrase in answer_lower for phrase in refusal_indicators)
            assert has_refusal, (
                f"Out-of-scope question not refused: {question}\n"
                f"Got: {r.json()['answer'][:200]}"
            )

