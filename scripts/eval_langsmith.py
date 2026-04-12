"""
LangSmith-Integrated Evaluation
────────────────────────────────
Runs the agent through a small batch of questions and pushes
ALL RAGAS metrics (context_recall, context_precision, faithfulness,
answer_relevancy) to LangSmith as feedback per query.

Use this for visualizing per-query metrics in the LangSmith UI.
For batch averages and JSON results, use eval_agentic.py instead.

Usage:
    uv run python -m scripts.eval_langsmith
"""

import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from scripts.llm_manager import LLMTask, get_llm
from agents.pipeline import PolicyAgentPipeline

load_dotenv()

GOLDEN_TEST_PATH = Path("data/golden_test_set/golden_test_set.json")

# Smaller batches for LangSmith deep evaluation
# 4 questions × 4 metrics = 16 LLM calls per run
LS_EVAL_START = 0
LS_EVAL_END = 4

# Metrics to run (add answer_relevancy and context_precision)
RAGAS_METRICS = [
    context_recall,
    context_precision,
    faithfulness,
    #answer_relevancy,
]


def load_questions():
    """Load a small batch for deep evaluation."""
    with open(GOLDEN_TEST_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    questions = [
        q for q in test_set
        if q["category"] != "out_of_scope"
    ]
    questions = questions[LS_EVAL_START:LS_EVAL_END]
    
    print(f"[OK] Loaded {len(questions)} questions for LangSmith eval")
    return questions


def run_agent_with_tracing(test_set):
    """Run agent and capture LangSmith run IDs."""
    print("\n[INIT] Building agent pipeline...")
    pipeline = PolicyAgentPipeline()
    pipeline.create_agent()
    print("[OK] Pipeline ready\n")

    results = []
    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"\n[{i}/{len(test_set)}] {question[:60]}")

        try:
            with collect_runs() as cb:
                graph = pipeline.create_agent()
                config = {
                    "configurable": {
                        "thread_id": f"ls_eval_{i}"
                    },
                    "metadata": {
                        "category": item["category"],
                        "difficulty": item.get("difficulty", "?"),
                        "eval_type": "langsmith_full_metrics",
                    },
                    "tags": [
                        "langsmith_eval",
                        f"category:{item['category']}",
                    ],
                }
                final_state = graph.invoke(
                    {"question": question},
                    config=config,
                )

            run_id = (
                str(cb.traced_runs[0].id)
                if cb.traced_runs else None
            )

            answer = final_state.get("answer", "")
            graded = final_state.get("graded_documents", [])
            contexts = [doc.page_content for doc in graded]
            if not contexts:
                contexts = ["(no documents retrieved)"]

            print(f"  [OK] Answer: {answer[:60]}...")
            print(f"  [OK] Run ID: {run_id}")

        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            answer = f"Error: {e}"
            contexts = ["(error)"]
            run_id = None

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
            "category": item["category"],
            "run_id": run_id,
        })

        time.sleep(3)

    return results


def evaluate_all_metrics(results):
    """Run all 4 RAGAS metrics on the results."""
    print("\n[EVAL] Running RAGAS with 4 metrics...")
    print(f"[EVAL] Estimated LLM calls: {len(results) * len(RAGAS_METRICS)}")

    llm = LangchainLLMWrapper(get_llm(LLMTask.RAGAS_JUDGE))

    # Local embeddings — required for answer_relevancy and context_precision
    # Without this, RAGAS defaults to OpenAI embeddings (which we don't use)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    )

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    scores = evaluate(
        dataset,
        metrics=RAGAS_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=1,
            max_retries=5,
            timeout=180,
        ),
    )

    df = scores.to_pandas()
    return df


def push_all_metrics_to_langsmith(results, scores_df):
    """Push all 4 metrics as feedback for each run."""
    client = Client()
    
    metric_columns = [
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    ]
    
    pushed_count = 0
    
    for result, (_, row) in zip(results, scores_df.iterrows()):
        if not result.get("run_id"):
            continue
        
        for metric in metric_columns:
            if metric not in row or row[metric] is None:
                continue
            
            try:
                client.create_feedback(
                    run_id=result["run_id"],
                    key=metric,
                    score=float(row[metric]),
                    comment=(
                        f"RAGAS evaluation, category: "
                        f"{result['category']}"
                    ),
                )
                pushed_count += 1
            except Exception as e:
                print(f"  [WARN] Failed: {metric} for {result['run_id']}: {e}")
    
    print(f"\n[LANGSMITH] Pushed {pushed_count} feedback entries")
    print(f"[LANGSMITH] {len(metric_columns)} metrics × "
          f"{len([r for r in results if r.get('run_id')])} runs")


def print_summary(scores_df):
    """Print average scores for all metrics."""
    print("\n" + "=" * 70)
    print("  LangSmith Evaluation — Per-Metric Averages")
    print("=" * 70)
    print(f"  {'Metric':<25}{'Score':>12}{'Status':>15}")
    print("  " + "-" * 60)
    
    targets = {
        "context_recall": 0.85,
        "context_precision": 0.80,
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
    }
    
    for metric, target in targets.items():
        if metric in scores_df.columns:
            avg = float(scores_df[metric].mean())
            status = "PASS" if avg >= target else f"target: {target}"
            print(f"  {metric:<25}{avg:>12.4f}{status:>15}")
    
    print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("  LangSmith Full-Metric Evaluation")
    print(f"  Questions: {LS_EVAL_START}-{LS_EVAL_END}")
    print("  Metrics: recall, precision, faithfulness, relevancy")
    print("=" * 70 + "\n")

    test_set = load_questions()
    results = run_agent_with_tracing(test_set)
    scores_df = evaluate_all_metrics(results)
    
    print_summary(scores_df)
    push_all_metrics_to_langsmith(results, scores_df)
    
    print("\n[DONE] Check LangSmith UI for per-query metrics:")
    print(f"  https://smith.langchain.com/projects/hragent")


if __name__ == "__main__":
    main()