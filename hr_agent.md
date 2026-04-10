
HR AI Copilot
VanaciRetain
Agentic RAG System for HR Teams
Architecture • Implementation • Deployment Plan

Multi-Agentic System with RAG, ML Prediction & MCP Server
LangGraph • LangChain • Groq • Supabase • Qdrant • AWS

Version:
2.0
Date:
March 2026
Target:
HR Teams, People Analysts, Data Scientists

1. Project Overview
HR teams spend significant time answering repetitive employee questions about policies, benefits, leave, and HR processes. This project builds an AI-powered HR Copilot (VanaciRetain) that automates these tasks while also providing data-driven insights about employee attrition and retention.

The system combines Retrieval Augmented Generation (RAG) with agentic orchestration, machine learning for attrition prediction, and HR workflow automation. The architecture mirrors enterprise AI systems used by companies such as Microsoft (Contoso HR Agent), while remaining simple enough to build incrementally.

Core Capabilities
  •  HR policy Q&A with agentic self-correcting RAG (retrieve, grade, re-query, generate)
  •  Employee attrition prediction using AutoGluon/XGBoost on IBM HR dataset
  •  Retention recommendations combining ML predictions with policy knowledge
  •  HR workflow automation via MCP server tools
  •  Full observability with LangSmith tracing and MLflow experiment tracking

1.1 Project Phases
The project is built in three phases, each delivering a working product:

Phase
Name
Description
Phase 1
HR Policy Assistant (RAG)
Agentic RAG chatbot answering HR questions using internal policy documents with self-correction loops
Phase 2
Attrition Risk Analysis
ML pipeline predicting which employees are likely to leave, with experiment tracking via MLflow
Phase 3
Retention Recommendation Engine
Combines ML predictions with policy retrieval to generate actionable retention recommendations

2. System Architecture
The system uses an agent-based architecture orchestrated with LangGraph. Three top-level agents handle different responsibilities, connected through a shared state and an intent router.

2.1 Agent Architecture
User Query
     │
Intent Router (LangGraph)
     │
 ┌────────────┬────────────┬────────────┐
Policy Agent    Workflow Agent    Analytics Agent
(Agentic RAG)   (MCP Tools)       (ML Model)
     │               │                  │
Vector DB +     Supabase          Supabase +
Qdrant          (actions)          ML Inference

2.2 Agent Responsibilities

Agent
Responsibility
Data Sources
Policy Agent
Handles HR policy queries using Agentic RAG. Internally uses LangGraph subgraph with corrective retrieval loops (retrieve → grade → re-query if needed → generate → check grounding).
Qdrant (vectors) + Supabase (metadata)
Workflow Agent
Handles HR actions like relocation requests, leave submissions, employee profile lookups. Exposes capabilities via MCP tools.
Supabase (employee data, workflow requests)
Analytics Agent
Analyzes employee data and predicts attrition risk. Combines ML model output with policy retrieval to generate retention recommendations.
Supabase (employee records) + ML model

3. Policy Agent — Agentic RAG Deep Dive
The Policy Agent is the core of Phase 1. Unlike naive RAG (retrieve once, generate once), it uses a LangGraph subgraph with self-correction loops. This is based on the Corrective RAG and Self-RAG patterns from recent research.

3.1 Internal LangGraph Flow
The Policy Agent is itself a LangGraph graph with 6 internal nodes and 2 conditional feedback loops:

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

3.2 LangGraph State
The shared state object flows between all nodes. Each node reads from and writes to this state:

class PolicyAgentState(MessagesState):
    # MessagesState provides "messages" for conversation memory
    transformed_query: str        # Optimized search query
    documents: List[Document]      # Retrieved chunks
    relevance_scores: List[float]  # Per-chunk grades (0-1)
    generation: str                # Generated answer
    is_grounded: bool              # Hallucination check result
    retry_count: int               # Re-query counter (max 2)

3.3 Conditional Edge Logic
	•	Grader → Retriever (re-query): Triggers when average relevance score < 0.6 AND retry_count < 2. The query transformer reformulates the search with different terms and loops back to retrieval.
	•	Grader → Generator (proceed): Triggers when average relevance ≥ 0.6 or retry_count has reached max. Passes only chunks scoring above 0.5 to the generator.
	•	Grounding Check → Generator (regenerate): If answer contains ungrounded claims, regenerate with stricter system prompt. Maximum 1 regeneration attempt.
	•	Router → END (out-of-scope): If query is classified as out_of_scope, return a polite refusal without invoking retrieval.

3.4 Conversation Memory (Multi-Turn)
The Policy Agent supports multi-turn conversations. When a user asks a follow-up like "What about part-time employees?" after asking about PTO, the agent uses conversation history to resolve ambiguous references.

The MessagesState automatically accumulates conversation turns. The transform_query node sees the full history and rewrites ambiguous follow-ups into standalone search queries. A sliding window of the last 5 turns (10 messages) is maintained to keep context manageable.

Turn 1:
  User: "How many PTO days do full-time employees get?"
  Bot:  "Full-time employees receive 15 PTO days per year..."

Turn 2:
  User: "What about part-time employees?"
  transform_query sees history, rewrites to:
  → "PTO vacation days policy for part-time employees"
  → Retrieves correct chunks → Generates contextual answer

4. Data Architecture
The system uses two databases — one for vector search (Qdrant) and one for structured data (Supabase/PostgreSQL). Both are used intentionally for different purposes and for learning both technologies.

4.1 Dual Database Design

Qdrant (Vector Database)
Supabase (PostgreSQL)
Purpose
Semantic search over HR documents. Stores embeddings for fast similarity search.
Structured data storage. Employee records, workflow requests, analytics results.
Stores
HR document embeddings, handbook chunks with metadata, policy content vectors
Employee records, attrition predictions, HR workflow requests, golden test set results
Used By
Policy Agent (retrieval), RAG evaluation pipeline
Workflow Agent (actions), Analytics Agent (ML predictions), all agents (metadata)
Cost
Free (self-hosted in Docker locally, or Qdrant Cloud free tier)
Free (Supabase free tier: 500 MB storage, unlimited API requests)

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

5. Technology Stack

Layer
Technology
LLM Inference
Groq API (Llama 3.1 70B / Llama 3.3 70B) — free tier
Embeddings
all-MiniLM-L6-v2 (sentence-transformers) — local, free, 384 dimensions
Vector Database
Qdrant (self-hosted in Docker locally, Qdrant Cloud for production)
Structured Database
Supabase (PostgreSQL) — free tier
Agent Framework
LangGraph + LangChain (Python 3.11+)
Backend API
FastAPI
MCP Server
FastAPI-based MCP server (Model Context Protocol)
ML Models
AutoGluon / XGBoost / LightGBM / Random Forest
RAG Evaluation
RAGAS (context recall, faithfulness, answer relevancy)
Tracing / Observability
LangSmith (integrated with LangChain/LangGraph)
MLOps
MLflow (experiment tracking, model versioning, model registry)
Deployment
AWS EC2 + S3 (Docker-based deployment)
CI/CD
GitHub Actions + Docker

6. HR Knowledge Base (RAG Data Sources)
The system uses a comprehensive 142-page employee handbook from Gallagher Franchise Solutions as the primary knowledge source. This handbook covers all major HR policy areas needed for realistic testing.

6.1 Primary Document
Source: Gallagher Franchise Solutions Employee Handbook (142 pages)
URL: franinsurance.com/media/pbka50b5/employeehandbookandguidelines.pdf

The handbook covers the following policy areas, making it ideal for diverse Q&A testing:
	•	Employment policies (at-will, classifications, hiring, onboarding)
	•	Compensation and payroll (pay schedules, overtime, deductions)
	•	Benefits (health insurance, dental, vision, life insurance, 401k)
	•	Leave policies (PTO, sick leave, FMLA, maternity/paternity, bereavement)
	•	Workplace conduct (harassment, drug policy, dress code, attendance)
	•	Safety and health (OSHA compliance, workplace safety, emergency procedures)
	•	Technology policies (internet usage, email, social media, data protection)
	•	Separation policies (resignation, termination, exit interviews)

6.2 Supplementary Documents
To test cross-document retrieval, additional focused documents will be added:
	•	HR FAQ document (common questions and quick answers)
	•	Relocation policy (specific procedures for employee relocation)
	•	Onboarding checklist (new hire process and requirements)

Total target size: 100–150 pages across all documents.

7. Document Preprocessing
The Gallagher handbook is a template with placeholders and blank fields. Before ingestion, these must be replaced with realistic values. This preprocessing step ensures the RAG system retrieves clean, realistic content.

7.1 Placeholder Replacement
All template placeholders are replaced with VanaciPrime-branded values:

Placeholder (Input)
Replaced With (Output)
Type
[ORGANIZATION NAME]
VanaciPrime
Company name
[Company Name]
VanaciPrime
Company name
[STATE]
California
Location
[CITY]
San Francisco
Location
[NUMBER OF HOURS]
40
Policy value
[HR CONTACT]
hr@vanaciprime.com
Contact
_____ (blank fields)
Realistic policy values
Policy specifics

7.2 Blank Field Population
The handbook contains blank underlines where specific policy values should go. These are filled with realistic values to make the golden test set meaningful:

Original (Input)
After Preprocessing (Output)
"Employees receive ___ vacation days per year"
"Employees receive 15 vacation days per year"
"Probationary period is ___ days"
"Probationary period is 90 days"
"Overtime is paid at ___ times the regular rate"
"Overtime is paid at 1.5 times the regular rate"
"[ORGANIZATION NAME] provides ___ sick days"
"VanaciPrime provides 10 sick days per year"
"Health insurance begins after ___ days"
"Health insurance begins after 30 days of employment"

7.3 Text Cleaning
After placeholder replacement, the extracted text goes through cleaning:
	•	Remove repeated headers/footers that appear on every page
	•	Normalize whitespace (multiple spaces, tabs, excessive line breaks)
	•	Preserve table structures by converting them to markdown format
	•	Extract and tag section headings for metadata enrichment
	•	Remove page numbers and decorative elements

Preprocessing Output
  •  A clean text file (or set of files) with all placeholders filled, blank fields populated with realistic values, and text cleaned for optimal chunking
  •  A metadata file mapping each section heading to its page range and policy category
  •  The original PDF preserved in S3 as the source of truth

8. RAG Pipeline Architecture

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

8.1 Chunking Strategy
	•	Chunk size: ~500 tokens with ~50 token overlap
	•	Splitter: RecursiveCharacterTextSplitter with separators ["\n\n", "\n", ". ", " "]
	•	Section-aware: Chunks respect section boundaries; a chunk should not span two unrelated policy sections
	•	Metadata enrichment: Each chunk carries: source_document, section_title, page_number, chunk_index, document_type
	•	Parent document retrieval: Store both small chunks (for precise matching) and parent sections (for richer context to the generator)

8.2 Embedding Model
Model: all-MiniLM-L6-v2 (sentence-transformers)
Dimensions: 384
Runs: Locally in the container (free, no API dependency)
Why this model: Fast inference on CPU, well-tested for retrieval tasks, 384 dimensions keeps Qdrant storage small, no external API costs.

9. Golden Test Set & Evaluation
A Golden Evaluation Dataset of 60+ Q&A pairs will be created from the preprocessed handbook. This is the ground truth for measuring RAG quality throughout development.

9.1 Question Categories
Category
Example Question
Tests
Factual
"How many sick leave days are allowed per year?"
Basic retrieval accuracy
Procedural
"What are the steps to file a harassment complaint?"
Multi-step extraction
Comparison
"What is the difference between short-term and long-term disability?"
Cross-section retrieval
Multi-hop
"If I exhaust my PTO, can I use sick leave for vacation?"
Reasoning across policies
Conditional
"After how many years does the PTO accrual rate increase?"
Nuanced conditional extraction
Out-of-scope
"What is VanaciPrime's stock price today?"
Appropriate refusal behavior

9.2 Evaluation Metrics
Metric
What It Measures
Target
Tool
Context Recall
% of golden answer facts present in retrieved chunks
≥ 85%
RAGAS
Faithfulness
% of generated claims grounded in retrieved context
≥ 90%
RAGAS
Answer Relevancy
How relevant the answer is to the original question
≥ 85%
RAGAS
Latency (P95)
95th percentile end-to-end response time
< 3 seconds
LangSmith
Hallucination Rate
% of responses containing ungrounded claims
< 5%
Manual + RAGAS

10. MCP Server Tools
The MCP server exposes structured tools to the agents. Agents interact with tools instead of direct database access, following the Model Context Protocol specification.

Tool
Description
Used By
search_hr_policy(query)
Semantic search over HR documents via Qdrant
Policy Agent
get_employee_profile(id)
Fetch employee details from Supabase
Workflow Agent
predict_attrition(id)
Run ML prediction for an employee
Analytics Agent
create_relocation_request(id)
Initiate relocation workflow in Supabase
Workflow Agent
list_documents()
List all indexed HR documents with metadata
Policy Agent

11. Phase 2 — Attrition Risk Analysis

11.1 Objective
Predict which employees are likely to leave the organization using machine learning, and surface these predictions through the Analytics Agent.

11.2 Dataset
Source: IBM HR Analytics Attrition Dataset (Kaggle)
URL: kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Target variable: Attrition (Yes/No)

Key features: Age, Department, Monthly Income, Overtime, Years at Company, Job Satisfaction, Promotion History, Work-Life Balance, Distance from Home.

11.3 ML Pipeline
HR Dataset (IBM Kaggle)
      │
Feature Engineering
      │
Model Training (AutoGluon / XGBoost / LightGBM)
      │
Evaluation (accuracy, precision, recall, F1)
      │
Model Registry (MLflow)
      │
Inference API (FastAPI endpoint)

11.4 Example Output
Employee: Sarah
Attrition Risk: 0.72 (HIGH)

Top Risk Factors:
  - Low job satisfaction (2/5)
  - High overtime (Yes)
  - No promotion in 4 years

12. Phase 3 — Retention Recommendations
The retention engine combines ML predictions from Phase 2 with policy retrieval from Phase 1 to generate actionable recommendations.

Employee Query (e.g., 'How can we retain John?')
      │
Analytics Agent
      │
Attrition Model → Risk score + top factors
      │
Policy Agent → Relevant policies (benefits, flexibility)
      │
Recommendation LLM → Actionable steps

Example Output:
John shows HIGH attrition risk (0.78) due to:
  - Overtime frequency and low work-life balance
  - No promotion in 3 years

Recommended Actions:
  1. Flexible working hours (per Remote Work Policy, Sec 4.2)
  2. Promotion review cycle (per Career Development Policy)
  3. Retention bonus discussion (per Compensation Policy, Sec 7.1)

13. Observability & Tracing

13.1 LangSmith
LangSmith is integrated directly with LangChain and LangGraph for full agent tracing:
	•	Track every agent step (route, retrieve, grade, generate)
	•	Track prompt inputs and outputs at each node
	•	Monitor per-step and end-to-end latency
	•	Debug RAG retrieval quality (which chunks were retrieved, scores)
	•	Evaluate prompt performance over time

13.2 MLflow
MLflow handles the ML side of observability:
	•	Experiment tracking for attrition model training runs
	•	Model versioning (track every model iteration with parameters and metrics)
	•	Model registry (promote best models to production)
	•	Prompt versioning (store and track prompt templates alongside model versions)

14. CI/CD Pipeline

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

Tools: GitHub Actions + Docker. Every push triggers tests, RAG evaluation against the golden test set, and model validation before deployment.

15. AWS Deployment Architecture

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

Local development uses Docker Compose with FastAPI + Qdrant + embedding model running locally, connecting to Supabase (cloud, free) and Groq API (free). The same Docker setup deploys to AWS EC2 with minimal config changes (just environment variables).

16. Repository Structure

hr-ai-copilot/
│
├── api/
│   └── main.py                  # FastAPI application
│
├── agents/
│   ├── policy_agent.py          # Agentic RAG (LangGraph subgraph)
│   ├── workflow_agent.py        # HR action execution
│   └── analytics_agent.py       # ML predictions + recommendations
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

17. Development Steps
The project follows a sequential build order. Each step produces a testable deliverable.

Step 1: Collect & Preprocess HR Documents  (Week 1)
•  Download Gallagher handbook PDF (142 pages)
•  Run preprocessing script: replace all placeholders with VanaciPrime values
•  Fill blank policy fields with realistic numbers (PTO days, probation period, etc.)
•  Clean extracted text: remove headers/footers, normalize whitespace, preserve tables as markdown
•  Store raw PDF in data/hr_documents/raw/ and cleaned text in data/hr_documents/processed/
•  Deliverable: Clean, VanaciPrime-branded HR document ready for ingestion

Step 2: Build RAG Pipeline  (Week 1-2)
•  Set up Qdrant (Docker locally) and Supabase project (free tier)
•  Implement document loader (PyMuPDF for PDF) and chunking (RecursiveCharacterTextSplitter, 500 tokens, 50 overlap)
•  Integrate all-MiniLM-L6-v2 for embeddings (sentence-transformers, local)
•  Ingest preprocessed handbook into Qdrant with section metadata
•  Build basic retriever with LangChain’s Qdrant integration
•  Test with simple queries to verify retrieval works
•  Deliverable: Working naive RAG pipeline (retrieve + generate) as baseline

Step 3: Create Golden Evaluation Dataset  (Week 2)
•  Read through the preprocessed handbook manually
•  Create 60+ Q&A pairs across all 6 categories (factual, procedural, comparison, multi-hop, conditional, out-of-scope)
•  For each pair, record: question, expected_answer, source_section, source_page, category, difficulty
•  Include 10-15 out-of-scope questions the system should refuse
•  Store as structured JSON in data/golden_test_set/
•  Run RAGAS evaluation on naive RAG baseline → record first metrics
•  Deliverable: Golden test set in version control + baseline RAGAS scores documented

Step 4: Build Agentic RAG (LangGraph)  (Week 3-4)
•  Implement LangGraph subgraph for Policy Agent with 6 nodes (route, transform, retrieve, grade, generate, check grounding)
•  Implement conditional edges for self-correction loops (re-query if relevance < 0.6, regenerate if hallucination detected)
•  Add conversation memory using MessagesState (sliding window of 5 turns)
•  Integrate Groq API (Llama 3.1 70B) as the LLM backend for all agent reasoning
•  Run RAGAS evaluation on agentic RAG → compare against naive baseline
•  Deliverable: Agentic RAG showing measurable improvement over baseline on golden test set

Step 5: Integrate LangSmith Tracing  (Week 4)
•  Connect LangSmith to the LangGraph pipeline
•  Verify all agent steps are traced: route decisions, retrieval results, grading scores, generation
•  Set up monitoring dashboards for latency and retrieval quality
•  Deliverable: Full observability into every agent decision

Step 6: Deploy Phase 1 to AWS  (Week 5)
•  Dockerize FastAPI backend + Qdrant + embedding model
•  Create docker-compose.yml for local dev and AWS deployment
•  Set up AWS EC2 instance, configure security groups, deploy containers
•  Connect to Supabase (cloud) and Groq API from EC2
•  Verify end-to-end flow works in production
•  Deliverable: Phase 1 accessible via HTTPS endpoint on AWS

Step 7: Train Attrition Prediction Model  (Week 6-7)
•  Download IBM HR Analytics dataset from Kaggle
•  Feature engineering and exploratory data analysis
•  Train models: AutoGluon (for auto-selection), XGBoost, LightGBM, Random Forest
•  Evaluate: accuracy, precision, recall, F1, AUC-ROC
•  Deliverable: Trained attrition model with documented performance metrics

Step 8: Track Experiments with MLflow  (Week 7)
•  Set up MLflow tracking server
•  Log all training runs: parameters, metrics, model artifacts
•  Register best model in MLflow model registry
•  Version prompts alongside model versions
•  Deliverable: MLflow experiment dashboard with all training runs tracked

Step 9: Deploy Phase 2 + Build Retention Agent  (Week 8-9)
•  Create inference API endpoint for attrition predictions
•  Build Analytics Agent that combines ML predictions with policy retrieval
•  Implement retention recommendation workflow (risk factors → relevant policies → actionable steps)
•  Build MCP server with all 5 tools (search_hr_policy, get_employee_profile, predict_attrition, create_relocation_request, list_documents)
•  Deliverable: Full multi-agent system with RAG + ML + recommendations

Step 10: Implement CI/CD Pipeline  (Week 10)
•  Set up GitHub Actions workflow
•  Automated: unit tests → RAG evaluation (golden test set) → model validation → Docker build → deploy
•  Configure deployment to AWS EC2
•  Run full regression suite on golden test set; ensure all metrics meet targets
•  Deliverable: Production-ready system with automated CI/CD

18. Success Criteria
The project is considered successful when all of the following criteria are met:

Criterion
Status
Context Recall ≥ 85% on golden test set
Target
Faithfulness ≥ 90% (no hallucinated policy info)
Target
End-to-end latency < 3s (P95)
Target
Agentic RAG outperforms naive RAG baseline on all RAGAS metrics
Target
Out-of-scope queries correctly refused (≥ 90%)
Target
Attrition model accuracy ≥ 76% (baseline from AutoGluon)
Target
MCP tools callable from all agents
Target
Deployed on AWS with CI/CD pipeline
Target
Full LangSmith tracing + MLflow experiment tracking
Target

19. Final System Capabilities
The completed system will support:

	•	HR policy Q&A with agentic self-correcting RAG and source citations
	•	Multi-turn conversations with conversation memory
	•	Employee attrition prediction with explainable risk factors
	•	Retention recommendations combining ML predictions with policy knowledge
	•	HR workflow automation via MCP server tools
	•	Full observability (LangSmith for agents, MLflow for ML models)
	•	Automated CI/CD with RAG quality gates
	•	AWS cloud deployment with Docker-based infrastructure

This project demonstrates a full production-style AI system combining Agentic RAG, multi-agent orchestration, ML models, MCP server integration, and cloud deployment — targeting HR teams, people analysts, and data scientists.

