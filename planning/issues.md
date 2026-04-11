# HR-Agent RAG Code Review Report

**Generated:** 2026-04-02  
**Scope:** `rag/` directory (baseline code only)  
**Reviewer:** code-reviewer skill (security + style + bugs)  

---

## 🟠 Summary

| Category | Count |
|----------|-------|
| **Security** | 0 Critical, 1 High |
| **Bugs** | 2 |
| **Code Quality** | 5 |
| **Style** | 3 |
| **Performance** | 2 |
| **Documentation** | 1 |

**Overall Assessment:** Code is functional but needs cleanup and defensive programming improvements before production.

---

## 🔴 Security Issues (1)

### HIGH: No Input Sanitization on Interactive Input

**File:** `rag/baseline_rag.py` (lines 54-56)  
**Severity:** HIGH

```python
while True:
    question = input("\nYour Question: ").strip()
```

**Issue:** Raw user input is passed directly to the LLM chain without any sanitization. While LangChain handles some escaping, there's no filtering for:
- Prompt injection attacks ("Ignore previous instructions and...")
- Very long inputs (DoS via context window exhaustion)
- Control characters or escape sequences

**Recommendation:**
```python
def sanitize_input(question: str) -> str:
    """Sanitize user input before processing."""
    # Limit length
    if len(question) > 1000:
        raise ValueError("Question too long (max 1000 chars)")
    
    # Basic prompt injection detection
    injection_patterns = [
        r"ignore\s+(?:previous|prior|all)",
        r"forget\s+(?:your|previous)",
        r"you\s+are\s+now",
        r"system\s*:",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            logging.warning(f"Potential prompt injection detected: {question[:50]}")
            return ""
    
    return question.strip()
```

---

## 🐛 Bugs (2)

### BUG-1: Division by Zero Risk in Empty Chunks

**File:** `rag/ingest_naive.py` (lines 88-92)  
**Severity:** MEDIUM

```python
lengths = [len(c.page_content) for c in chunks]
print(f"[OK] Created {len(chunks)} chunks")
print(f"Avg: {sum(lengths) // len(lengths)} chars")  # Line 91: Division by zero if chunks empty
print(f"Min: {min(lengths)}, Max: {max(lengths)} chars")  # Line 92: min/max on empty list
```

**Issue:** If `split_documents()` returns empty chunks (edge case with empty input), `sum(lengths) // len(lengths)` raises `ZeroDivisionError`, and `min()`/`max()` raise `ValueError`.

**Fix:**
```python
if chunks:
    lengths = [len(c.page_content) for c in chunks]
    print(f"[OK] Created {len(chunks)} chunks")
    print(f"Avg: {sum(lengths) // len(lengths)} chars")
    print(f"Min: {min(lengths)}, Max: {max(lengths)} chars")
else:
    print("[WARN] No chunks created")
```

---

### BUG-2: Unused Import Statement

**File:** `rag/ingest_naive.py` (line 19)  
**Severity:** LOW

```python
from pydoc import doc  # Never used
```

**Issue:** Dead code. `pydoc.doc` is imported but never referenced.

**Fix:** Remove line 19.

---

### BUG-3: Multiple Unused Imports

**File:** `rag/baseline_evaluation.py` (lines 13, 17, 24)  
**Severity:** LOW

```python
from ragas import embeddings      # Line 13: Never used
import pandas                     # Line 17: Never used (pandas imported as 'pandas', not 'pd')
from openai import OpenAI         # Line 24: Never used (using Groq, not OpenAI)
```

**Fix:** Remove these three imports.

---

## ⚠️ Code Quality Issues (5)

### QUAL-1: No Error Handling for File I/O

**File:** `rag/ingest_naive.py` (line 32)  
**Severity:** MEDIUM

```python
def load_policies() -> list[Document]:
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)
```

**Issue:** No try/except for:
- JSON decode errors (malformed policies.json)
- Permission errors
- File encoding issues

**Same issue in:** `rag/baseline_evaluation.py` line 35

**Fix:**
```python
def load_policies() -> list[Document]:
    policies_path = PROCESSED_DIR / "policies.json"
    
    if not policies_path.exists():
        raise FileNotFoundError(f"policies.json not found at {policies_path}")
    
    try:
        with open(policies_path, "r", encoding="utf-8") as f:
            policies = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {policies_path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error in {policies_path}: {e}")
    
    # Validate structure
    if not isinstance(policies, list):
        raise ValueError(f"Expected list, got {type(policies)}")
    
    return policies
```

---

### QUAL-2: No Validation of Retrieved Documents

**File:** `rag/baseline_rag.py` (function `format_docs`)  
**Severity:** MEDIUM

```python
def format_docs(docs: list) -> str:
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        policy = doc.metadata.get("policy_name", "Unknown")
        # ...
```

**Issue:** No validation that `docs` is a list of Document objects. If retriever fails, this could receive unexpected types.

**Fix:**
```python
def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents found."
    
    # Validate document type
    for doc in docs:
        if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
            logger.warning(f"Invalid document type: {type(doc)}")
            continue
    # ... rest of function
```

---

### QUAL-3: Hardcoded Sleep Without Exponential Backoff

**File:** `rag/baseline_evaluation.py` (line 104)  
**Severity:** LOW

```python
results.append({...})
time.sleep(3)  # Fixed 3 second delay
```

**Issue:** Fixed sleep is inefficient. If rate limit is 10 req/min, this is fine, but if limit is hit, script doesn't adapt.

**Recommendation:** Implement retry with exponential backoff or use a rate limiter library.

---

### QUAL-4: No Logging Configuration

**File:** `rag/baseline_rag.py`, `rag/ingest_naive.py`, `rag/baseline_evaluation.py`  
**Severity:** LOW

**Issue:** Using `print()` statements instead of `logging` module. Production code should use structured logging.

**Fix:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace prints
logger.info("Created %d chunks", len(chunks))
logger.warning("No documents found")
```

---

### QUAL-5: Global State in Module-Level Variables

**File:** Multiple files  
**Severity:** LOW

```python
# rag/baseline_rag.py
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
PROMPT = ChatPromptTemplate.from_messages([...])  # Global prompt
```

**Issue:** Module-level configuration makes testing difficult and can cause issues with reloads.

**Recommendation:** Use dependency injection or factory functions:
```python
def create_baseline_chain(model: str | None = None) -> Runnable:
    llm_model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    # ... create chain with parameters
```

---

## 🎨 Style Issues (3)

### STYLE-1: Inconsistent Whitespace

**File:** `rag/retriever.py` (line 81)  
**Severity:** LOW

```python
    return vector_store.as_retriever(
        search_type = search_type,              # Spaces around =
        search_kwargs={"k":k},                  # No spaces
    )
     # <-- Trailing whitespace
```

**Fix:** Use consistent formatting (spaces around operators):
```python
search_type=search_type,
search_kwargs={"k": k}
```

---

### STYLE-2: Trailing Whitespace in File

**File:** `rag/retriever.py` (line 98)  
**Severity:** LOW

Line 98 has trailing whitespace characters.

---

### STYLE-3: Import Order

**File:** `rag/baseline_evaluation.py` (lines 1-30)  
**Severity:** LOW

Imports are not grouped by standard library / third-party / local.

**Current:**
```python
import json
import os
from ragas import evaluate  # Third-party mixed with stdlib
from pathlib import Path
```

**Should be:**
```python
# Standard library
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Third-party
from datasets import Dataset
from langchain_groq import ChatGroq

# Local
from rag.baseline_rag import baseline_rag_chain
from rag.retriever import get_retriever
```

---

## ⚡ Performance Issues (2)

### PERF-1: Synchronous Evaluation with Sleep

**File:** `rag/baseline_evaluation.py` (lines 71-104)  
**Severity:** MEDIUM

```python
for i, item in enumerate(test_set, 1):
    # ...
    answer = chain.invoke(question)  # Blocking call
    time.sleep(3)  # Blocking sleep
```

**Issue:** Sequential processing with fixed delays. For 30 questions at 3s each = 90s minimum, not accounting for LLM latency.

**Recommendation:** Add TODO comment for future async implementation:
```python
# TODO: Convert to async with asyncio.gather for batch processing
```

---

### PERF-2: Embedding Model Reloaded Per Process

**File:** `rag/retriever.py` (line 28)  
**Severity:** LOW (already mitigated)

```python
@lru_cache(maxsize=1)
def get_embeddings():
```

**Note:** Current implementation correctly uses `@lru_cache`, but there's no cache invalidation mechanism for long-running processes.

**Recommendation:** Document that this pattern assumes the embedding model doesn't change during process lifetime.

---

## 📝 Documentation Issues (1)

### DOC-1: Missing Docstring for Main Functions

**File:** `rag/baseline_rag.py` (function `main`)  
**Severity:** LOW

```python
def main():
    print("\n" + "=" * 60)
```

**Issue:** Missing docstring explaining:
- What the CLI does
- How to exit (quit/exit/q)
- Expected input format

**Fix:**
```python
def main():
    """
    Run interactive RAG CLI.
    
    Allows users to ask HR policy questions interactively.
    Type 'quit', 'exit', or 'q' to exit.
    """
```

---

## 🔧 Recommended Action Plan

### Immediate (Before Next Commit)
1. **Fix BUG-1** (division by zero) - prevents runtime crash
2. **Remove unused imports** (BUG-2, BUG-3) - cleaner code
3. **Fix STYLE-1** (trailing whitespace) - trivial fix

### Short Term (Before Production)
4. **Fix QUAL-1** (file I/O error handling)
5. **Address HIGH security** (input sanitization)
6. **Replace print with logging** (QUAL-4)
7. **Add document validation** (QUAL-2)

### Nice to Have
8. Refactor to use dependency injection
9. Add async batch processing
10. Add comprehensive input validation

---

## Files Affected Summary

| File | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| `rag/baseline_rag.py` | 0 | 1 | 2 | 2 |
| `rag/ingest_naive.py` | 0 | 0 | 2 | 3 |
| `rag/retriever.py` | 0 | 0 | 0 | 2 |
| `rag/baseline_evaluation.py` | 0 | 0 | 1 | 4 |

---

## Verification Commands

Run these to check fixes:

```bash
# Ruff for style
ruff check rag/ --fix

# MyPy for types
mypy rag/

# Bandit for security
bandit -r rag/

# Find unused imports
ruff check rag/ --select F401
```

---

**End of Report**
