# Policy Agent Spec
## VanaciRetain — Agentic RAG Architecture

**Version:** 2.0 | **Date:** March 2026

---

## 1. Purpose

The Policy Agent is the core of Phase 1. It answers HR policy questions using a self-correcting LangGraph subgraph (retrieve → grade → re-query → generate → check grounding). Unlike naive RAG, it never returns an answer it can't ground in the retrieved context.

This document covers the Policy Agent internals, the RAG pipeline, the knowledge base, preprocessing, and the golden test set used for evaluation.

---

## 2. Internal LangGraph Flow

The Policy Agent is a LangGraph subgraph with 6 nodes and 2 conditional feedback loops:

```
User Query
     │
     ▼
┌─────────────────┐
│  Route Query      │ ── Is this an HR question?
└────────┬────────┘    No → Return 'out of scope'
         │ Yes
         ▼
┌─────────────────┐
│ Transform Query   │ ── Rewrite for better retrieval
└────────┬────────┘    Uses conversation history for context
         │
         ▼
┌─────────────────┐
│   Retrieve         │ ── Search Qdrant, get top-k chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────┐
│ Grade Documents   │─No─▶│ Rewrite Query      │
│ (relevance OK?)   │       │ (different terms)  │
└────────┬────────┘       └────────┬─────────┘
         │ Yes                      │
         │               ◀─────────┘  (loop back, max 2)
         ▼
┌─────────────────┐
│ Generate Answer   │ ── Synthesize from graded chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Grounding   │ ── Is answer supported by context?
└────────┬────────┘
    Yes  │    No → Regenerate (stricter prompt, max 1 retry)
         ▼
  Final Response (with source citations)
```

---

## 3. LangGraph State

```python
class PolicyAgentState(MessagesState):
    # MessagesState provides "messages" for conversation memory
    transformed_query: str        # Optimized search query
    documents: List[Document]     # Retrieved chunks
    relevance_scores: List[float] # Per-chunk grades (0-1)
    generation: str               # Generated answer
    is_grounded: bool             # Hallucination check result
    retry_count: int              # Re-query counter (max 2)
```

---

## 4. Conditional Edge Logic

- **Grader → Retriever (re-query):** Triggers when average relevance score < 0.6 AND `retry_count` < 2. The query transformer reformulates the search with different terms.
- **Grader → Generator (proceed):** Triggers when average relevance ≥ 0.6 or `retry_count` has reached max. Passes only chunks scoring above 0.5.
- **Grounding Check → Generator (regenerate):** If answer contains ungrounded claims, regenerate with stricter system prompt. Maximum 1 regeneration attempt.
- **Router → END (out-of-scope):** If query is classified as `out_of_scope`, return a polite refusal without invoking retrieval.

---

## 5. Conversation Memory (Multi-Turn)

The agent supports multi-turn conversations. `MessagesState` accumulates conversation turns. `transform_query` sees the full history and rewrites ambiguous follow-ups into standalone search queries. A sliding window of the last 5 turns (10 messages) is maintained.

```
Turn 1:
  User: "How many PTO days do full-time employees get?"
  Bot:  "Full-time employees receive 15 PTO days per year..."

Turn 2:
  User: "What about part-time employees?"
  transform_query sees history, rewrites to:
  → "PTO vacation days policy for part-time employees"
  → Retrieves correct chunks → Generates contextual answer
```

---

## 6. RAG Pipeline Architecture

```
HR Documents (preprocessed)
       │
Document Loader (PDF/DOCX/Markdown)
       │
Chunking (RecursiveCharacterTextSplitter)
       │
Embedding (all-MiniLM-L6-v2, 384 dims)
       │
Vector Database (Qdrant)
       │
Retriever (top-k with MMR diversity)
       │
Agentic RAG Pipeline (grade → re-query → generate)
       │
LLM Response (Groq / Llama 3)
```

### 6.1 Chunking Strategy

- Chunk size: ~500 tokens with ~50 token overlap
- Splitter: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " "]`
- Section-aware: chunks respect section boundaries
- Metadata per chunk: `source_document`, `section_title`, `page_number`, `chunk_index`, `document_type`
- Parent document retrieval: store both small chunks (precise matching) and parent sections (richer context)

### 6.2 Embedding Model

- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Dimensions:** 384
- **Runs:** locally in container (no API dependency)
- **Why:** fast CPU inference, well-tested for retrieval, 384 dims keeps Qdrant storage small

---

## 7. HR Knowledge Base

**Primary source:** Gallagher Franchise Solutions Employee Handbook (142 pages)
URL: franinsurance.com/media/pbka50b5/employeehandbookandguidelines.pdf

Policy areas covered:
- Employment policies (at-will, classifications, hiring, onboarding)
- Compensation and payroll (pay schedules, overtime, deductions)
- Benefits (health insurance, dental, vision, life insurance, 401k)
- Leave policies (PTO, sick leave, FMLA, maternity/paternity, bereavement)
- Workplace conduct (harassment, drug policy, dress code, attendance)
- Safety and health (OSHA compliance, emergency procedures)
- Technology policies (internet usage, email, social media, data protection)
- Separation policies (resignation, termination, exit interviews)

**Supplementary documents (planned):**
- HR FAQ document
- Relocation policy
- Onboarding checklist

Total target size: 100–150 pages.

---

## 8. Document Preprocessing

The Gallagher handbook is a template with placeholders. All must be replaced before ingestion.

### 8.1 Placeholder Replacement

| Placeholder | Replaced With | Type |
|---|---|---|
| `[ORGANIZATION NAME]` | VanaciPrime | Company name |
| `[Company Name]` | VanaciPrime | Company name |
| `[STATE]` | California | Location |
| `[CITY]` | San Francisco | Location |
| `[NUMBER OF HOURS]` | 40 | Policy value |
| `[HR CONTACT]` | hr@vanaciprime.com | Contact |
| `_____` (blank fields) | Realistic policy values | Policy specifics |

### 8.2 Blank Field Population

| Original | After Preprocessing |
|---|---|
| "Employees receive ___ vacation days per year" | "Employees receive 15 vacation days per year" |
| "Probationary period is ___ days" | "Probationary period is 90 days" |
| "Overtime is paid at ___ times the regular rate" | "Overtime is paid at 1.5 times the regular rate" |
| "VanaciPrime provides ___ sick days" | "VanaciPrime provides 10 sick days per year" |
| "Health insurance begins after ___ days" | "Health insurance begins after 30 days of employment" |

### 8.3 Text Cleaning

- Remove repeated headers/footers
- Normalize whitespace
- Preserve table structures as markdown
- Extract and tag section headings for metadata
- Remove page numbers and decorative elements

**Preprocessing outputs:**
- Clean text file with all placeholders filled
- Metadata file mapping section headings to page ranges and policy categories
- Original PDF preserved in S3 as source of truth

---

## 9. Golden Test Set & Evaluation

60+ Q&A pairs created from the preprocessed handbook.

### 9.1 Question Categories

| Category | Example | Tests |
|---|---|---|
| Factual | "How many sick leave days are allowed per year?" | Basic retrieval accuracy |
| Procedural | "What are the steps to file a harassment complaint?" | Multi-step extraction |
| Comparison | "What is the difference between short-term and long-term disability?" | Cross-section retrieval |
| Multi-hop | "If I exhaust my PTO, can I use sick leave for vacation?" | Reasoning across policies |
| Conditional | "After how many years does the PTO accrual rate increase?" | Nuanced conditional extraction |
| Out-of-scope | "What is VanaciPrime's stock price today?" | Appropriate refusal behavior |

### 9.2 Evaluation Metrics

| Metric | What It Measures | Target | Tool |
|---|---|---|---|
| Context Recall | % of golden answer facts in retrieved chunks | ≥ 85% | RAGAS |
| Faithfulness | % of generated claims grounded in context | ≥ 90% | RAGAS |
| Answer Relevancy | How relevant the answer is to the question | ≥ 85% | RAGAS |
| Latency (P95) | 95th percentile end-to-end response time | < 3 seconds | LangSmith |
| Hallucination Rate | % of responses with ungrounded claims | < 5% | Manual + RAGAS |

---

## 10. MCP Tools (Policy Agent)

| Tool | Description |
|---|---|
| `search_hr_policy(query)` | Semantic search over HR documents via Qdrant |
| `list_documents()` | List all indexed HR documents with metadata |

Full MCP server spec (all tools) is in `deployment_spec.md`.

---

## 11. Build Steps (Phase 1)

**Step 1 — Collect & Preprocess HR Documents (Week 1)**
- Download Gallagher handbook PDF (142 pages)
- Run preprocessing: replace placeholders with VanaciPrime values, fill blank fields, clean text
- Store raw PDF in `data/hr_documents/raw/`, cleaned text in `data/hr_documents/processed/`
- Deliverable: clean, VanaciPrime-branded HR document ready for ingestion

**Step 2 — Build RAG Pipeline (Week 1-2)**
- Set up Qdrant (Docker locally) and Supabase project
- Implement document loader (PyMuPDF) and chunking (500 tokens, 50 overlap)
- Integrate all-MiniLM-L6-v2 embeddings (local)
- Ingest preprocessed handbook into Qdrant with section metadata
- Deliverable: working naive RAG pipeline as baseline

**Step 3 — Create Golden Evaluation Dataset (Week 2)**
- Create 60+ Q&A pairs across all 6 categories
- Each pair: `question`, `expected_answer`, `source_section`, `source_page`, `category`, `difficulty`
- Run RAGAS on naive RAG baseline → record first metrics
- Deliverable: golden test set + baseline RAGAS scores

**Step 4 — Build Agentic RAG (Week 3-4)**
- Implement LangGraph subgraph with 6 nodes and conditional self-correction loops
- Add conversation memory (MessagesState, sliding window of 5 turns)
- Integrate Groq API (Llama 3.1 70B)
- Run RAGAS on agentic RAG vs naive baseline
- Deliverable: agentic RAG with measurable improvement over baseline

**Step 5 — Integrate LangSmith Tracing (Week 4)**
- Connect LangSmith to the LangGraph pipeline
- Verify all agent steps are traced (route decisions, retrieval results, grading scores)
- Deliverable: full observability into every agent decision
