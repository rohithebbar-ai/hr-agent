"""
Agentic RAG Evaluation
──────────────────────
Runs golden test set against the LangGraph agentic pipeline.
Compares results to baseline and policy-aware scores.
 
Usage:
    # Set EVAL_START and EVAL_END below, then run:
    uv run python -m rag.eval_agentic
"""

import json
import time
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics import context_recall, faithfulness
from ragas.run_config import RunConfig

from scripts.llm_manager import LLMTask, get_llm
from agents.pipeline import PolicyAgentPipeline
from agents.schemas import PolicyAgentState
from langchain_core.tracers.context import collect_runs
from langsmith import Client


load_dotenv()

# ── Config ──
GOLDEN_TEST_PATH = Path("data/golden_test_set/golden_test_set.json")
RESULTS_DIR = Path("data/golden_test_set/agentic_eval_results")

# Run in batches of 10 to stay under Groq rate limits
EVAL_START = 0
EVAL_END = 50


# Previous scores average for comparison (from baseline + policy-aware runs)
BASELINE_SCORES = {
    "context_recall": 0.65,
    "faithfulness": 0.47,
}
 
POLICY_AWARE_SCORES = {
    "context_recall": 0.69,
    "faithfulness": 0.54,
}
 
TARGETS = {
    "context_recall": 0.85,
    "faithfulness": 0.90,
}

def load_test_set() -> list:
    """Load golden test set, excluding out-of-scope questions."""
    with open(GOLDEN_TEST_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    questions = [
        q for q in test_set
        if q["category"] != "out_of_scope"
    ]
    questions = questions[EVAL_START:EVAL_END]

    print(f"[OK] Evaluating questions {EVAL_START}-{EVAL_END}")
    print(f"[OK] Loaded {len(questions)} questions")
    return questions


def generate_agentic_answers(test_set: list) -> list:
    """Run each question through the agentic LangGraph pipeline."""
    print("\n[INIT] Building agent pipeline")
    pipeline = PolicyAgentPipeline()
    pipeline.create_agent() # pre-compile the graph
    print(f"[OK] Pipeline ready\n")

    results = []
    total = len(test_set)

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"\n[{i:2d}/{total}] {question[:100]}")

        try:
            # Run through agent — use unique thread_id per question
            # so previous turns don't pollute the conversation
            # Wrap in collect_runs to capture LangSmith run ID
            with collect_runs() as cb:
                graph = pipeline.create_agent()
                config = {
                    "configurable": {
                        "thread_id": f"eval_{EVAL_START}_{i}"
                    },
                    "metadata": {
                        "category": item["category"],
                        "eval_batch": f"{EVAL_START}-{EVAL_END}",
                    },
                    "tags": [
                        "evaluation",
                        f"category:{item['category']}",
                    ],
                }
                final_state = graph.invoke(
                    {"question": question},
                    config=config,
                )
            
            # Extract the top-level run ID
            run_id =(
                str(cb.traced_runs[0].id)
                if cb.traced_runs else None
            )

            answer = final_state.get("answer", "")

            # Extract the contexts from graded documents
            graded = final_state.get("graded_documents", [])
            contexts = [doc.page_content for doc in graded]

            # Fallback: if grading filtered everything, use raw documents
            if not contexts:
                raw_docs = final_state.get("documents", [])
                contexts = [doc.page_content for doc in raw_docs]

            # Final fallback: provide at least one empty context for RAGAS
            if not contexts:
                contexts = ["(no documents retrieved)"]

        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            answer = f"Error during agent execution: {e}"
            contexts = ["(error)"]
            run_id = None
 
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
            "category": item["category"],
            "difficulty": item.get("difficulty", "unknown"),
            "run_id": run_id,
        })

        # Pacing for rate limits — agentic pipeline already uses
        # multiple LLM calls per question, so be conservative
        time.sleep(3)
 
    return results

def push_ragas_to_langsmith(results, ragas_scores_df):
    """Push per-question RAGAS scores back to LangSmith as feedback."""
    client = Client()
    pushed = 0
    
    for result, (_, row) in zip(results, ragas_scores_df.iterrows()):
        if not result.get("run_id"):
            continue
        
        try:
            # Push context_recall as feedback
            if "context_recall" in row and row["context_recall"] is not None:
                client.create_feedback(
                    run_id=result["run_id"],
                    key="context_recall",
                    score=float(row["context_recall"]),
                    comment=f"RAGAS evaluation, {result['category']}",
                )
            
            # Push faithfulness as feedback
            if "faithfulness" in row and row["faithfulness"] is not None:
                client.create_feedback(
                    run_id=result["run_id"],
                    key="faithfulness",
                    score=float(row["faithfulness"]),
                    comment=f"RAGAS evaluation, {result['category']}",
                )
            
            pushed += 1
        except Exception as e:
            print(f"[WARN] Failed to push feedback for run {result['run_id']}: {e}")
    print(f"[LANGSMITH] Pushed RAGAS feedback for {pushed} runs")

def evaluate_with_ragas(results: list) -> dict:
    """Run RAGAS metrics on the generated answers."""
    print("\n[EVAL] Running RAGAS evaluation")

    # Use the dedicated RAGAS judge LLM (uses second API key)
    llm = LangchainLLMWrapper(get_llm(LLMTask.RAGAS_JUDGE))

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    scores = evaluate(
        dataset,
        metrics=[context_recall, faithfulness],
        llm=llm,
        run_config=RunConfig(
            max_workers=1,
            max_retries=5,
            timeout=120,
        ),
    )

    df = scores.to_pandas()
    # Return both averages AND the per-question dataframe
    return {
        "averages": {
            "context_recall": float(df["context_recall"].mean()),
            "faithfulness": float(df["faithfulness"].mean()),
        },
        "per_question": df,
    }
def save_results(results: list, scores: dict):
    """Save detailed results and scores with metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # Save detailed results
    results_path = (
        RESULTS_DIR / f"agentic_results_q{EVAL_START}-{EVAL_END}_{ts}.json"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
 
    # Save scores with metadata
    scores_path = (
        RESULTS_DIR / f"agentic_scores_q{EVAL_START}-{EVAL_END}_{ts}.json"
    )

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "type": "agentic_langgraph",
            "num_questions": len(results),
            "question_range": f"{EVAL_START}-{EVAL_END}",
            "pipeline_features": [
                "query_decomposition",
                "batched_grading",
                "corrective_loops",
                "grounding_check",
                "multi_llm_routing",
            ],
            "scores": scores,
            "baseline_scores": BASELINE_SCORES,
            "policy_aware_scores": POLICY_AWARE_SCORES,
        }, f, indent=2, default=str)
 
    print(f"\n[OK] Results: {results_path}")
    print(f"[OK] Scores:  {scores_path}")
 

def print_comparison(scores: dict):
    """Print three-way comparison: baseline → policy-aware → agentic."""
    print("\n" + "=" * 80)
    print("  COMPARISON: Baseline → Policy-Aware → Agentic")
    print("=" * 80)
    print(
        f"  {'Metric':<20s}"
        f"{'Baseline':>12s}"
        f"{'Policy-Aware':>14s}"
        f"{'Agentic':>12s}"
        f"{'Δ (PA→A)':>11s}"
        f"{'Target':>9s}"
    )
    print("  " + "─" * 76)
 
    for metric in ["context_recall", "faithfulness"]:
        base = BASELINE_SCORES.get(metric, 0)
        pa = POLICY_AWARE_SCORES.get(metric, 0)
        ag = scores.get(metric, 0)
        target = TARGETS.get(metric, 0)
 
        delta = ag - pa
        sign = "+" if delta >= 0 else ""
        status = " PASS" if ag >= target else ""
 
        print(
            f"  {metric:<20s}"
            f"{base:>12.4f}"
            f"{pa:>14.4f}"
            f"{ag:>12.4f}"
            f"{sign}{delta:>10.4f}"
            f"{target:>8.2f}{status}"
        )
 
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  Agentic RAG Evaluation")
    print("  LangGraph Policy Agent with query decomposition")
    print(f"  Questions: {EVAL_START}-{EVAL_END}")
    print("=" * 60 + "\n")
 
    test_set = load_test_set()
 
    print("[RUN] Generating answers through agent")
    results = generate_agentic_answers(test_set)
 
    eval_output = evaluate_with_ragas(results)
    scores = eval_output["averages"]
    per_question = eval_output["per_question"]

    push_ragas_to_langsmith(results, per_question)
 
    print_comparison(scores)
    save_results(results, scores)
 
    print("[DONE] Agentic evaluation complete.") 
 
if __name__ == "__main__":
    main()
 