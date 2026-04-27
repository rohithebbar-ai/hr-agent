"""
Guardrails
──────────
Input validation, prompt injection detection, output sanitization,
and PII redaction for the API layer.

These run BEFORE and AFTER the LLM pipeline to prevent abuse
and protect sensitive data.
"""

import re


# ══════════════════════════════════════════════════
# INPUT GUARDRAILS
# ══════════════════════════════════════════════════

# Known prompt injection patterns

# Known prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"act\s+as\s+(?:a|an)\s+",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"system\s*prompt",
    r"reveal\s+your\s+(instructions?|prompts?|rules?)",
    r"what\s+(?:are|is)\s+your\s+(system\s+)?prompt",
    r"show\s+me\s+your\s+(instructions?|prompts?)",
    r"repeat\s+(?:the\s+)?(?:above|previous)\s+(?:text|instructions?)",
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"```\s*system",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Check if the input contains prompt injection attempts.

    Returns:
        (is_safe, reason) — True if safe, False if injection detected.
    """
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(text):
            return False, f"potential prompt injection detected"

    return True, ""

def validate_input(question: str) -> tuple[bool, str]:
    """
    Run all input validations on the questions.
    Returns
        (is_valid, error_message) - True if valid, false with reason
    """
    # Check empty
    if not question or not question.strip():
        return False, "Question cannot be empty"

    # Check length
    if len(question) > 2000:
        return False, (
            f"question too long ({len(question)} chars)"
            f"Max limit is 2000 chars"
        )

    # Check if minimum meaningful length
    if len(question.strip()) < 2:
        return False, "question too short"

    # Check for prompt injection
    is_safe, reason = check_prompt_injection(question)
    if not is_safe:
        return False, reason
    
    return True, ""

    # Check for excessive special characters (possible encoding attack)
    """
    special_ratio = sum(
        1 for c in question
        if not c.isalnum() and not c.isspace() and c not in ".,?!'-()/"
    )/ max(len(question))

    if special_ratio > 0.3:
        return False, "question contains too many special characters
    """

# ══════════════════════════════════════════════════
# OUTPUT GUARDRAILS
# ══════════════════════════════════════════════════

# PII patterns for redaction in logs
PII_PATTERNS = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
     "[EMAIL_REDACTED]"),
    # Phone numbers (various formats)
    (re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
     "[PHONE_REDACTED]"),
    # SSN patterns
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     "[SSN_REDACTED]"),
    # Employee IDs (common patterns like EMP-12345)
    (re.compile(r"\b[Ee][Mm][Pp][-_]?\d{4,6}\b"),
     "[EMPLOYEE_ID_REDACTED]"),
]

def redact_pii_logging(text: str) -> str:
    """
    Redact PII from text before writing to logs
    The original response to user is NOT modified - 
    only logged version is redacted
    """
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted

# URL pattern for stripping hallucinated links
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)

def sanitize_output(answer: str) -> str:
    """
    Clean the LLM output before sending to the user.
    Removes hallucinated URLs and enforces length limits.
    """
    # Strip any URLs (LLM might hallucinate links)
    sanitized = URL_PATTERN.sub("[link removed]", answer)

    # Cap response length (prevent runaway generation)
    max_length = 3000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized