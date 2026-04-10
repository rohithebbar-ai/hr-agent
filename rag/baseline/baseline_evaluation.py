"""
Baseline RAG Evaluation with RAGAS
───────────────────────────────────
Runs the golden test set against the naive RAG chain.
Measures context recall, faithfulness, and answer relevancy.
 
Usage:
    uv run python rag/evaluation.py
"""
import json
import os
from pathlib import Path
#from ragas import embeddings
from datasets import Dataset
from ragas import evaluate
from rag.baseline.baseline_rag import baseline_rag_chain
from rag.retriever import get_retriever
from ragas.run_config import RunConfig
import pandas 
from ragas.metrics import context_recall, faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from datetime import datetime
from dotenv import load_dotenv
import time
from openai import OpenAI

load_dotenv()

GOLDEN_TEST_PATH = Path("data/golden_test_set/golden_test_set.json")
RESULTS_DIR = Path("data/golden_test_set/baseline_eval_results")

EVAL_LIMIT = 10 
EVAL_START = 40
EVAL_END = 50


def load_golden_test_set() -> list:
    """ Load golden test set """
    with open(GOLDEN_TEST_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Filter out-of scope questions
    hr_questions = [
        q for q in test_set
        if q["category"] != "out_of_scope"
    ]
    hr_questions = hr_questions[EVAL_START:EVAL_END] 
    print(f"[OK] Loaded {len(hr_questions)} HR Questions")
    return hr_questions

def generate_baseline_answers(test_set: list) -> list:
    """ Run RAG chain on all test questions"""
    chain = baseline_rag_chain()
    retriever = get_retriever(search_type="similarity", k = 3)

    results = []
    total = len(test_set)

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"[{i:2d}/{total}] {question[:55]}")

        # Get retrieved content
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # Get generated answer
        answer = chain.invoke(question)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
            "category": item["category"],
            "difficulty": item["difficulty"],
        })
        time.sleep(3) 

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
    llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant")
    )

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
        max_retries=3,       # retry on timeout
        timeout=60,          # seconds per call
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
    """Save detailed results and scores."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # Save detailed results
    results_path = RESULTS_DIR / f"baseline_results_q{EVAL_START}-{EVAL_END}_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
 
    # Save scores
    scores_path = RESULTS_DIR / f"baseline_scores_q{EVAL_START}-{EVAL_END}_{timestamp}.json"
    scores_with_meta = {
        "timestamp": timestamp,
        "type": "baseline_naive_rag",
        "num_questions": len(results),
        "retrieval": "similarity_k5",
        "chunking": "blind_500_50",
        "scores": scores,
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_with_meta, f, indent=2, default=str)
 
    print(f"[OK] Results: {results_path}")
    print(f"[OK] Scores:  {scores_path}")
 
 
def print_scores(scores: dict):
    """Print formatted scores with pass/fail against targets."""
    targets = {
        "context_recall": 0.85,
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
    }
 
    print("\n" + "=" * 55)
    print("  BASELINE RAGAS SCORES (Naive RAG)")
    print("=" * 55)
    for metric, value in scores.items():
        target = targets.get(metric, 0)
        if isinstance(value, (int, float)):
            status = "PASS" if value >= target else "BELOW TARGET"
            print(
                f"{metric:25s}: {value:.4f}"
                f"(target: {target:.2f}) [{status}]"
            )
 
def main():
    print("\n" + "=" * 60)
    print("  Baseline RAG Evaluation with RAGAS")
    print("  Naive RAG | similarity search | blind chunks")
    print("=" * 60 + "\n")
 
    # 1. Load golden test set
    test_set = load_golden_test_set()
 
    # 2. Generate answers with naive RAG
    print("[RUN] Generating baseline answers\n")
    results = generate_baseline_answers(test_set)
 
    # 3. Evaluate with RAGAS
    scores = evaluate_results_ragas(results)
 
    # 4. Print and save
    print_scores(scores)
    save_results(results, scores)
 
    print("[DONE] Baseline evaluation complete.\n")
 
 
if __name__ == "__main__":
    main()
 
