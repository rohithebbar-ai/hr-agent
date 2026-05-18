from api.guardrails.guardrails import (
    check_prompt_injection,
    redact_pii_logging,
    sanitize_output,
    validate_input,
)

class TestInputValidation:
    def test_valid_question(self):
        is_valid, msg = validate_input("How many vacation days?")
        assert is_valid is True, f"Validation failed! The error message was: {msg}"