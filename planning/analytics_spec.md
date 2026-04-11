# Analytics Spec
## VanaciRetain — Attrition ML & Retention Recommendations

**Version:** 2.0 | **Date:** March 2026

---

## 1. Purpose

This document covers Phase 2 (attrition prediction ML model) and Phase 3 (retention recommendation engine). These are handled by the Analytics Agent, which is distinct from the Policy Agent (RAG) and Workflow Agent (MCP actions).

---

## 2. Phase 2 — Attrition Risk Analysis

### 2.1 Objective

Predict which employees are likely to leave the organization using machine learning, and surface these predictions through the Analytics Agent.

### 2.2 Dataset

- **Source:** IBM HR Analytics Attrition Dataset (Kaggle)
- **URL:** kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Target variable:** `Attrition` (Yes/No)
- **Key features:** Age, Department, Monthly Income, Overtime, Years at Company, Job Satisfaction, Promotion History, Work-Life Balance, Distance from Home

### 2.3 ML Pipeline

```
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
```

### 2.4 Models

| Model | Role |
|---|---|
| AutoGluon | Auto model selection and ensembling |
| XGBoost | Gradient boosting baseline |
| LightGBM | Fast gradient boosting alternative |
| Random Forest | Interpretable baseline |

Target accuracy: ≥ 76% (AutoGluon baseline).

### 2.5 Example Output

```
Employee: Sarah
Attrition Risk: 0.72 (HIGH)

Top Risk Factors:
  - Low job satisfaction (2/5)
  - High overtime (Yes)
  - No promotion in 4 years
```

---

## 3. Phase 3 — Retention Recommendations

The retention engine combines ML predictions from Phase 2 with policy retrieval from the Policy Agent to generate actionable recommendations.

### 3.1 Flow

```
Employee Query (e.g., 'How can we retain John?')
      │
Analytics Agent
      │
Attrition Model → Risk score + top factors
      │
Policy Agent → Relevant policies (benefits, flexibility)
      │
Recommendation LLM → Actionable steps
```

### 3.2 Example Output

```
John shows HIGH attrition risk (0.78) due to:
  - Overtime frequency and low work-life balance
  - No promotion in 3 years

Recommended Actions:
  1. Flexible working hours (per Remote Work Policy, Sec 4.2)
  2. Promotion review cycle (per Career Development Policy)
  3. Retention bonus discussion (per Compensation Policy, Sec 7.1)
```

---

## 4. MCP Tool (Analytics Agent)

| Tool | Description |
|---|---|
| `predict_attrition(id)` | Run ML prediction for a given employee |

Full MCP server spec (all tools) is in `deployment_spec.md`.

---

## 5. Observability — MLflow

MLflow handles the ML side of observability:
- Experiment tracking for all attrition model training runs
- Model versioning (parameters, metrics, artifacts per run)
- Model registry (promote best models to production)
- Prompt versioning (store prompt templates alongside model versions)

---

## 6. Build Steps (Phase 2 & 3)

**Step 7 — Train Attrition Prediction Model (Week 6-7)**
- Download IBM HR Analytics dataset from Kaggle
- Feature engineering and exploratory data analysis
- Train models: AutoGluon, XGBoost, LightGBM, Random Forest
- Evaluate: accuracy, precision, recall, F1, AUC-ROC
- Deliverable: trained attrition model with documented performance metrics

**Step 8 — Track Experiments with MLflow (Week 7)**
- Set up MLflow tracking server
- Log all training runs: parameters, metrics, model artifacts
- Register best model in MLflow model registry
- Version prompts alongside model versions
- Deliverable: MLflow experiment dashboard with all runs tracked

**Step 9 — Deploy Phase 2 + Build Retention Agent (Week 8-9)**
- Create inference API endpoint for attrition predictions
- Build Analytics Agent combining ML predictions with policy retrieval
- Implement retention recommendation workflow (risk factors → relevant policies → actionable steps)
- Deliverable: Analytics Agent with full attrition + recommendation capability
