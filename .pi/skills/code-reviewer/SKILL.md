---
name: code-reviewer
description: Comprehensive Python and JavaScript/TypeScript code reviewer that checks for security vulnerabilities, bugs, style violations, performance issues, and best practices. Use when asked to review code, check code quality, audit security, or find bugs in Python/JavaScript/TypeScript files.
license: MIT
compatibility: Python 3.11+, Node.js 18+, requires ruff, bandit, mypy, semgrep
---

# Code Reviewer Skill

A comprehensive code review assistant that performs multi-layer analysis on Python and JavaScript/TypeScript codebases.

## Capabilities

- **Security Analysis**: Detects vulnerabilities (SQL injection, XSS, hardcoded secrets, unsafe eval, etc.)
- **Static Analysis**: Bug detection, type checking, unused imports/variables
- **Style & Formatting**: PEP 8, Black, Ruff compliance
- **Performance**: Identifies inefficient patterns, N+1 queries, memory issues
- **Best Practices**: Architecture patterns, error handling, logging, documentation

## Setup

Install required tools (run once):

```bash
# Python tools
pip install ruff bandit mypy semgrep safety

# JavaScript/TypeScript tools (if needed)
npm install -g eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
```

## Quick Usage

```bash
# Review a single file
./scripts/review.py path/to/file.py

# Review entire directory
./scripts/review.py src/

# Security-only review
./scripts/security-check.py path/to/file.py

# Full codebase audit
./scripts/full-audit.py --path src/ --output report.json
```

## Review Process

When asked to review code:

1. **Identify the scope**: Single file, module, or entire codebase?
2. **Check file types**: Python (.py), JavaScript (.js), TypeScript (.ts)
3. **Run appropriate tools**:
   - Security: bandit (Python), semgrep (all)
   - Style: ruff check
   - Types: mypy (Python)
   - Bugs: pylint, eslint (JS/TS)
4. **Analyze results**: Categorize by severity (critical, warning, info)
5. **Provide actionable feedback**: Include line numbers, fix suggestions, and references

## Severity Levels

- **CRITICAL**: Security vulnerabilities, potential crashes, data loss
- **HIGH**: Likely bugs, significant performance issues
- **MEDIUM**: Style violations, minor optimizations, missing tests
- **LOW**: Documentation, formatting, nitpicks

## Output Format

```
## Review Summary
- Files reviewed: N
- Critical: X | High: Y | Medium: Z | Low: W
- Security issues: N
- Type errors: N

## Critical Issues
1. [File:Line] Issue description
   Fix: Suggested solution

## High Priority
...

## Recommendations
...
```

See [references/](references/) for detailed security checklists and style guides.