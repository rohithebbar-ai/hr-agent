# Deployment Spec
## VanaciRetain — Infrastructure, CI/CD & AWS Architecture

**Version:** 2.0 | **Date:** March 2026

---

## 1. Purpose

This document covers the full technology stack, data architecture, MCP server, CI/CD pipeline, AWS deployment, and repository structure for VanaciRetain. It is infrastructure-focused — agent internals are in `policy_agent_spec.md` and `analytics_spec.md`.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| LLM Inference | Groq API (Llama 3.1 70B / Llama 3.3 70B) — free tier |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) — local, free, 384 dims |
| Vector Database | Qdrant (self-hosted in Docker locally, Qdrant Cloud for production) |
| Structured Database | Supabase (PostgreSQL) — free tier |
| Agent Framework | LangGraph + LangChain (Python 3.11+) |
| Backend API | FastAPI |
| MCP Server | FastAPI-based MCP server (Model Context Protocol) |
| ML Models | AutoGluon / XGBoost / LightGBM / Random Forest |
| RAG Evaluation | RAGAS (context recall, faithfulness, answer relevancy) |
| Tracing / Observability | LangSmith (integrated with LangChain/LangGraph) |
| MLOps | MLflow (experiment tracking, model versioning, model registry) |
| Deployment | AWS EC2 + S3 (Docker-based) |
| CI/CD | GitHub Actions + Docker |

---

## 3. Data Architecture

The system uses two databases — one for vector search (Qdrant) and one for structured data (Supabase/PostgreSQL).

### 3.1 Dual Database Design

| | Qdrant (Vector DB) | Supabase (PostgreSQL) |
|---|---|---|
| **Purpose** | Semantic search over HR documents | Structured data storage |
| **Stores** | HR document embeddings, handbook chunks, policy vectors | Employee records, attrition predictions, workflow requests, golden test set results |
| **Used By** | Policy Agent (retrieval), RAG evaluation pipeline | Workflow Agent (actions), Analytics Agent (ML), all agents (metadata) |
| **Cost** | Free (self-hosted Docker or Qdrant Cloud free tier) | Free (Supabase free tier: 500 MB storage) |

```
             HR AI Copilot
                  │
              FastAPI
                  │
           LangGraph Agents
                  │
     ┌────────────┴────────────┐
     │                         │
 Vector Database          Postgres
   (Qdrant)              (Supabase)
     │                         │
 HR Documents          Employee Data
 Policy Embeddings     Attrition Scores
                       Workflow Requests
```

---

## 4. MCP Server — All Tools

The MCP server exposes structured tools to all agents. Agents use tools instead of direct database access, following the Model Context Protocol spec.

| Tool | Description | Used By |
|---|---|---|
| `search_hr_policy(query)` | Semantic search over HR documents via Qdrant | Policy Agent |
| `get_employee_profile(id)` | Fetch employee details from Supabase | Workflow Agent |
| `predict_attrition(id)` | Run ML prediction for an employee | Analytics Agent |
| `create_relocation_request(id)` | Initiate relocation workflow in Supabase | Workflow Agent |
| `list_documents()` | List all indexed HR documents with metadata | Policy Agent |

---

## 5. Observability — LangSmith

LangSmith is integrated with LangChain and LangGraph for full agent tracing:
- Track every agent step (route, retrieve, grade, generate)
- Track prompt inputs and outputs at each node
- Monitor per-step and end-to-end latency
- Debug RAG retrieval quality (which chunks were retrieved, scores)
- Evaluate prompt performance over time

---

## 6. CI/CD Pipeline

```
Code Commit (GitHub)
      │
Unit Tests
      │
RAG Evaluation (run golden test set)
      │
Model Validation (check accuracy thresholds)
      │
Docker Build
      │
Deployment to AWS
```

Every push to main triggers: unit tests → RAG evaluation against golden test set → model validation → Docker build → deploy to AWS EC2.

---

## 7. AWS Deployment Architecture

```
Frontend (Streamlit / React)
         │
   API Gateway
         │
  FastAPI Backend
         │
  LangGraph Agents
         │
  ───────────────
  │             │
Vector DB     Postgres
(Qdrant)     (Supabase)
  │
S3 (Documents)
```

Local development uses Docker Compose with FastAPI + Qdrant + embedding model running locally, connecting to Supabase (cloud) and Groq API. The same Docker setup deploys to AWS EC2 — only environment variables change.

---

## 8. Repository Structure

```
hr-ai-copilot/
│
├── api/
│   └── main.py                  # FastAPI application
│
├── agents/
│   ├── policy_agent.py          # Agentic RAG (LangGraph subgraph)
│   ├── workflow_agent.py        # HR action execution
│   └── analytics_agent.py      # ML predictions + recommendations
│
├── rag/
│   ├── document_loader.py       # PDF/DOCX parsing
│   ├── preprocessor.py          # Placeholder replacement + cleaning
│   ├── embeddings.py            # all-MiniLM-L6-v2 integration
│   ├── retriever.py             # Qdrant retrieval logic
│   └── evaluation.py            # RAGAS evaluation runner
│
├── models/
│   └── attrition_model.py       # ML training + inference
│
├── mcp/
│   └── hr_tools.py              # MCP tool definitions
│
├── mlops/
│   └── mlflow_tracking.py       # Experiment tracking setup
│
├── data/
│   ├── hr_documents/            # Raw + preprocessed HR docs
│   ├── golden_test_set/         # Evaluation Q&A pairs
│   └── attrition_dataset/       # IBM HR dataset
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── aws_setup.md
│
└── tests/                       # Unit + integration tests
```

---

## 9. Build Steps (Deployment)

**Step 6 — Deploy Phase 1 to AWS (Week 5)**
- Dockerize FastAPI backend + Qdrant + embedding model
- Create `docker-compose.yml` for local dev and AWS deployment
- Set up AWS EC2 instance, configure security groups, deploy containers
- Connect to Supabase (cloud) and Groq API from EC2
- Deliverable: Phase 1 accessible via HTTPS endpoint on AWS

**Step 10 — Implement CI/CD Pipeline (Week 10)**
- Set up GitHub Actions workflow
- Automated: unit tests → RAG evaluation → model validation → Docker build → deploy
- Configure deployment to AWS EC2
- Run full regression suite on golden test set; ensure all metrics meet targets
- Deliverable: production-ready system with automated CI/CD

---

## 10. Success Criteria

| Criterion | Target |
|---|---|
| Context Recall ≥ 85% on golden test set | Target |
| Faithfulness ≥ 90% (no hallucinated policy info) | Target |
| End-to-end latency < 3s (P95) | Target |
| Agentic RAG outperforms naive RAG on all RAGAS metrics | Target |
| Out-of-scope queries correctly refused (≥ 90%) | Target |
| Attrition model accuracy ≥ 76% (AutoGluon baseline) | Target |
| MCP tools callable from all agents | Target |
| Deployed on AWS with CI/CD pipeline | Target |
| Full LangSmith tracing + MLflow experiment tracking | Target |
