import pytest
from pydantic import ValidationError
from agents.schemas import RouteQuery, GroundingCheck

class TestRouteQuery:
    def test_uppercase(self):
        assert RouteQuery(category="CHAT").category == "CHAT"

    def test_lowercase_normalized(self):
        assert RouteQuery(category="chat").category == "CHAT"

    def test_rag(self):
        assert RouteQuery(category="RAG").category == "RAG"

    def test_invalid_rejected(self):
        with pytest.raises(ValidationError):
            RouteQuery(category="INVALID")


class TestGroundingCheck:
    def test_grounded(self):
        assert GroundingCheck(is_grounded="grounded").is_grounded == "grounded"

    def test_refusal_is_grounded(self):
        result = GroundingCheck(is_grounded="I don't have enough information")
        assert result.is_grounded == "grounded"