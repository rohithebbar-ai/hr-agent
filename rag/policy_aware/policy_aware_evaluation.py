"""
Policy-Aware RAG Evaluation
───────────────────────────
Runs golden test set against policy-aware RAG chain.
Compares results to baseline scores.
 
Usage:
    uv run python -m rag.eval_policy_aware
"""
 
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rag.config import golden_test_dir, golden_test_path
from rag.policy_aware.policy_aware_rag import build_chain
from rag.retriever import (
        COLLECTION_POLICY_AWARE,
        get_rerank_retriever,
        get_retriever
    )
from scripts.llm_manager import LLMTask, get_llm
import pandas 
from ragas.metrics import context_recall, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate

load_dotenv()

# Match the same number of questions used in baseline
# for a fair comparison. Set None for all questions.
EVAL_START = 0  
EVAL_END = 10


# Baseline scores (from your baseline run)
BASELINE_SCORES = {
    "context_recall": 0.55,
    "faithfulness": 0.63,
}
 
TARGETS = {
    "context_recall": 0.85,
    "faithfulness": 0.90,
}

RESULTS_DIR = Path("data/golden_test_set/policy_aware_eval_results")
 
def load_test_set() -> list:
    """Load golden test set, excluding out-of-scope."""
    with open(golden_test_path(), "r", encoding="utf-8") as f:
        test_set = json.load(f)

    questions = [
        q for q in test_set
        if q["category"] != "out_of_scope"
    ]
    
    questions = questions[EVAL_START:EVAL_END]
    print(f"[OK] Evaluating questions {EVAL_START}-{EVAL_END}")
    print(f"[OK] Loaded {len(questions)} questions")
    return questions


def generate_answers(test_set: list) -> list:
    """Generate answers using policy-aware RAG chain """
    chain = build_chain()
    retriever = get_retriever(
        collection=COLLECTION_POLICY_AWARE,
        search_type="mmr",
        k=5,
    )

    results = []
    total = len(test_set)

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"[{i:2d}/{total}] {question[:55]}")

        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        answer = chain.invoke(question)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
            "category": item["category"],
            "difficulty": item["difficulty"],
        })

        # small delay to avoid rate limits
        time.sleep(4)

    return results

def evaluate_results_ragas(results:list) -> dict:
    """
    Run RAGAS metrics on the results.
 
    Metrics:
    - context_recall: do retrieved chunks contain the answer?
    - faithfulness: is the generated answer grounded in context?
    - answer_relevancy: is the answer relevant to the question?
    """
    # Configure RAGAS to use llm
    llm = LangchainLLMWrapper(get_llm(LLMTask.RAGAS_JUDGE))

    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(data)

    print("\n[EVAL] Running RAGAS evaluation")
    score = evaluate(
        dataset,
        metrics=[context_recall, faithfulness],
        llm=llm,
        run_config=RunConfig(
        max_workers=1,       # one call at a time — no parallelism
        max_retries=5,       # retry on timeout
        timeout=120,          # seconds per call
    ),
    )
    df = score.to_pandas()
    score_dict = {
        "context_recall": df["context_recall"].mean(),
        "faithfulness": df["faithfulness"].mean(),
    }

    if not score_dict:
        raise RuntimeError("RAGAS returned no scores.")

    return score_dict

def save_results(results: list, scores: dict):
    """Save results and scores with metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    results_path = RESULTS_DIR / f"policy_aware_results_q{EVAL_START}-{EVAL_END}_{ts}_rerank.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
 
    scores_path = RESULTS_DIR / f"policy_aware_scores_q{EVAL_START}-{EVAL_END}_{ts}_rerank.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "type": "policy_aware_mmr",
            "num_questions": len(results),
            "retrieval": "mmr_k5",
            "chunking": "policy_aware_threshold_1000",
            "scores": scores,
            "baseline_scores": BASELINE_SCORES,
        }, f, indent=2, default=str)
 
    print(f"[OK] Results: {results_path}")
    print(f"[OK] Scores:  {scores_path}")

def print_comparison(scores: dict):
    """Print side-by-side comparison with baseline."""
    print("\n" + "=" * 68)
    print("  COMPARISON: Naive Baseline → Policy-Aware + MMR")
    print("=" * 68)
    print(
        f"{'Metric':<25s}"
        f"{'Baseline':>10s}"
        f"{'Policy-Aware':>14s}"
        f"{'Delta':>10s}"
        f"{'Target':>9s}"
    )
    print("  " + "─" * 64)
 
    for metric in ["context_recall", "faithfulness"]:
        base = BASELINE_SCORES.get(metric, 0)
        new = scores.get(metric, 0)
        target = TARGETS.get(metric, 0)
 
        if isinstance(new, (int, float)):
            delta = new - base
            sign = "+" if delta >= 0 else ""
            status = "✓" if new >= target else ""
            print(
                f"{metric:<25s}"
                f"{base:>10.4f}"
                f"{new:>14.4f}"
                f"{sign}{delta:>9.4f}"
                f"{target:>8.2f}{status}"
            )
 
    print("=" * 68)
    print()
 
 
def main():
    print("\n" + "=" * 60)
    print("Policy-Aware + MMR Evaluation")
    print(f"Baseline: CR={BASELINE_SCORES['context_recall']:.2f} "
          f"F={BASELINE_SCORES['faithfulness']:.2f}")
    print("=" * 60 + "\n")
 
    test_set = load_test_set()
 
    print("[RUN] Generating answers\n")
    results = generate_answers(test_set)
 
    scores = evaluate_results_ragas(results)
 
    print_comparison(scores)
    save_results(results, scores)

    print("[DONE] Policy-aware evaluation complete.")
 
if __name__ == "__main__":
    main()


    


