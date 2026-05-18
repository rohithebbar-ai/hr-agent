from scripts.preprocess_handbook import (
    apply_context_aware_replacements,
    fix_parenthesis_placeholders,
)


class TestContextReplacements:
    def test_lunch_break(self):
        text = "allowed an unpaid [insert number]-minute lunch break"
        result = apply_context_aware_replacements(text)
        assert "30-minute lunch break" in result

    def test_rest_periods(self):
        text = "Two paid rest periods of [insert number] minutes"
        result = apply_context_aware_replacements(text)
        assert "15 minutes" in result

    def test_fulltime_hours(self):
        text = "work at least [insert number] hours per week"
        result = apply_context_aware_replacements(text)
        assert "40 hours" in result


class TestParenthesisPlaceholders:
    def test_insert_replaced(self):
        text = "permitted (insert amount here) vacation"
        result = fix_parenthesis_placeholders(text)
        assert "(insert amount here)" not in result