"""
Reranker Benchmark CLI
───────────────────────────────────────────────────────
Compares flashrank reranker models using the golden test set.
Also benchmarks raw hybrid retrieval (no reranking) as a baseline.

Relevance is judged by matching key terms from expected_answer against
chunk text — NOT section names — so this works for any document or
handbook, not just the Gallagher template.

Metrics: Hit@1, Hit@3, MRR@8 (non-out-of-scope questions only).

Usage:
    python scripts/eval_reranker.py
    python scripts/eval_reranker.py --diverse          # 2 questions per category (10 total)
    python scripts/eval_reranker.py --models ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-12-v2
    python scripts/eval_reranker.py --limit 20 --tenant vanaciprime
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AVAILABLE_MODELS = [
    "ms-marco-TinyBERT-L-2-v2",   # default ~4 MB, fast
    "ms-marco-MiniLM-L-12-v2",    # better quality ~33 MB
    "ms-marco-electra-small",      # ~54 MB
    "rank-T5-flan",                # ~80 MB
]

# Words too generic to be useful relevance signals in HR policy text
STOP_WORDS = {
    "employee", "employees", "employer", "company", "vanaciprime", "organization",
    "policy", "policies", "shall", "their", "which", "these", "those", "about",
    "after", "during", "before", "either", "where", "while", "under", "within",
    "without", "between", "through", "however", "include", "including", "following",
    "provided", "based", "period", "right", "rights", "other", "work", "working",
    "time", "days", "hours", "leave", "with", "from", "that", "this", "will",
    "have", "been", "they", "them", "also", "both", "each", "such", "when",
    "then", "upon", "may", "must", "should", "would", "could", "does", "does",
    "employment", "handbook", "written", "notice", "provide", "request",
}


def extract_key_terms(expected_answer: str) -> dict:
    """
    Extract discriminative numbers and keywords from expected_answer.

    Numbers like "15", "90", "12", "1.5" are highly discriminative in HR
    policy text — a chunk about vacation containing "15" is almost certainly
    the right chunk. Keywords back this up when numbers are absent.
    """
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', expected_answer)
    words = re.findall(r'\b[a-zA-Z]{5,}\b', expected_answer.lower())
    keywords = list({w for w in words if w not in STOP_WORDS})
    return {"numbers": numbers, "keywords": keywords}


def is_relevant(chunk_text: str, expected_answer: str, source_section: str = "") -> bool:
    """
    Document-agnostic relevance check: does this chunk contain enough of
    the expected answer to be considered a hit?

    Rules:
    - Any number from expected_answer found in chunk → relevant
    - 2+ keywords from expected_answer found in chunk → relevant
    - Multi-hop (source_section=Multiple): 1+ keyword match is enough,
      since no single chunk will contain the full multi-policy answer
    """
    terms = extract_key_terms(expected_answer)
    text_lower = chunk_text.lower()

    num_matches = sum(
        1 for n in terms["numbers"]
        if re.search(rf'\b{re.escape(n)}\b', chunk_text)
    )
    kw_matches = sum(1 for kw in terms["keywords"] if kw in text_lower)

    if source_section == "Multiple":
        return num_matches >= 1 or kw_matches >= 1

    return num_matches >= 1 or kw_matches >= 2


def retrieve_candidates(query: str, tenant_id: str):
    """Get hybrid search candidates without reranking."""
    import os
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector, Prefetch, FusionQuery, Fusion
    from fastembed import SparseTextEmbedding
    from sentence_transformers import SentenceTransformer

    if not hasattr(retrieve_candidates, "_dense"):
        print("  Loading dense model...", flush=True)
        retrieve_candidates._dense = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Loading sparse model...", flush=True)
        retrieve_candidates._sparse = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        retrieve_candidates._client = (
            QdrantClient(url=url, api_key=api_key, timeout=30) if api_key
            else QdrantClient(url=url, timeout=30)
        )

    dense_vec = retrieve_candidates._dense.encode(query, normalize_embeddings=True).tolist()
    sparse_result = list(retrieve_candidates._sparse.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )
    tenant_filter = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])
    client = retrieve_candidates._client

    results = client.query_points(
        collection_name="hr_documents",
        prefetch=[
            Prefetch(query=dense_vec, using="dense", filter=tenant_filter, limit=40),
            Prefetch(query=sparse_vec, using="sparse", filter=tenant_filter, limit=40),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=40,
        with_payload=True,
    )
    return results.points


_ranker_cache: dict = {}


def rerank_candidates(candidates, query: str, model_name: str):
    """Rerank candidates with the given flashrank model. Returns ordered list of payloads."""
    if model_name == "retrieval_only":
        return [c.payload for c in candidates]

    from flashrank import Ranker, RerankRequest
    if model_name not in _ranker_cache:
        print(f"  [cache] loading {model_name}...", flush=True)
        _ranker_cache[model_name] = Ranker(model_name=model_name)
    ranker = _ranker_cache[model_name]
    passages = [{"id": i, "text": c.payload.get("text", "")} for i, c in enumerate(candidates)]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [candidates[r["id"]].payload for r in reranked]


def compute_metrics(
    ranked_payloads: list[dict],
    expected_answer: str,
    source_section: str,
    k_values: tuple = (1, 3, 8),
) -> dict:
    """
    Compute Hit@k and MRR by checking chunk text against expected_answer.
    Works for any document — no section name coupling.
    """
    hits = {k: 0 for k in k_values}
    rr = 0.0
    first_hit_rank = None
    first_hit_section = ""

    for rank, payload in enumerate(ranked_payloads, start=1):
        chunk_text = payload.get("text", "")
        if is_relevant(chunk_text, expected_answer, source_section):
            for k in k_values:
                if rank <= k:
                    hits[k] = 1
            if rr == 0.0:
                rr = 1.0 / rank
                first_hit_rank = rank
                first_hit_section = payload.get("section_path", "")
            break

    return {
        f"hit@{k}": hits[k] for k in k_values
    } | {
        "rr": rr,
        "first_hit_rank": first_hit_rank,
        "first_hit_section": first_hit_section,
    }


def run_benchmark(models: list[str], questions: list[dict], tenant_id: str) -> dict:
    results: dict[str, list[dict]] = {m: [] for m in models}

    total = len(questions)
    for i, q in enumerate(questions, 1):
        query = q["question"]
        expected_answer = q["expected_answer"]
        source_section = q["source_section"]
        category = q["category"]
        difficulty = q["difficulty"]

        terms = extract_key_terms(expected_answer)
        print(f"\n[{i:02d}/{total}] {query[:70]}", flush=True)
        print(f"  expected: {expected_answer[:80]}", flush=True)
        print(f"  key terms: numbers={terms['numbers']} keywords={terms['keywords'][:6]}", flush=True)

        try:
            candidates = retrieve_candidates(query, tenant_id)
        except Exception as e:
            print(f"  [ERROR] retrieval failed: {e}", flush=True)
            for m in models:
                results[m].append({"hit@1": 0, "hit@3": 0, "hit@8": 0, "rr": 0.0,
                                   "category": category, "difficulty": difficulty,
                                   "first_hit_rank": None, "first_hit_section": ""})
            continue

        for model_name in models:
            t0 = time.perf_counter()
            try:
                ranked = rerank_candidates(candidates, query, model_name)
                elapsed = time.perf_counter() - t0
                metrics = compute_metrics(ranked, expected_answer, source_section)
                metrics.update({"category": category, "difficulty": difficulty, "latency_s": elapsed})
                results[model_name].append(metrics)
                hit_info = (
                    f"rank={metrics['first_hit_rank']} section='{metrics['first_hit_section'][:40]}'"
                    if metrics["first_hit_rank"] else "NO HIT"
                )
                print(
                    f"  {model_name}: hit@1={metrics['hit@1']} hit@3={metrics['hit@3']} "
                    f"MRR={metrics['rr']:.3f} ({elapsed:.2f}s) → {hit_info}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}", flush=True)
                results[model_name].append({"hit@1": 0, "hit@3": 0, "hit@8": 0, "rr": 0.0,
                                            "category": category, "difficulty": difficulty,
                                            "latency_s": 0.0, "first_hit_rank": None,
                                            "first_hit_section": ""})

    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "hit@1":         round(sum(r["hit@1"] for r in results) / n, 3),
        "hit@3":         round(sum(r["hit@3"] for r in results) / n, 3),
        "hit@8":         round(sum(r["hit@8"] for r in results) / n, 3),
        "mrr@8":         round(sum(r["rr"]    for r in results) / n, 3),
        "avg_latency_s": round(sum(r.get("latency_s", 0) for r in results) / n, 3),
        "n": n,
    }


def print_report(models: list[str], results: dict[str, list[dict]]):
    categories = sorted({r["category"] for rows in results.values() for r in rows})

    print("\n" + "=" * 70)
    print("  RERANKER BENCHMARK RESULTS  (answer-based relevance)")
    print("=" * 70)

    col_w = 22
    header = f"{'Model':<{col_w}} {'Hit@1':>6} {'Hit@3':>6} {'Hit@8':>6} {'MRR@8':>6} {'Lat(s)':>7} {'N':>4}"

    print("\nOverall")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for m in models:
        agg = aggregate(results[m])
        print(f"{m:<{col_w}} {agg['hit@1']:>6.3f} {agg['hit@3']:>6.3f} {agg['hit@8']:>6.3f} {agg['mrr@8']:>6.3f} {agg['avg_latency_s']:>7.3f} {agg['n']:>4}")

    for cat in categories:
        print(f"\n  Category: {cat}")
        print("─" * len(header))
        print(header)
        print("─" * len(header))
        for m in models:
            cat_rows = [r for r in results[m] if r["category"] == cat]
            agg = aggregate(cat_rows)
            if agg:
                print(f"{m:<{col_w}} {agg['hit@1']:>6.3f} {agg['hit@3']:>6.3f} {agg['hit@8']:>6.3f} {agg['mrr@8']:>6.3f} {agg['avg_latency_s']:>7.3f} {agg['n']:>4}")

    print("\n" + "=" * 70)

    if "retrieval_only" in models and len(models) > 1:
        base_agg = aggregate(results["retrieval_only"])
        print("\n  Delta vs retrieval_only (hybrid RRF baseline):")
        for m in models:
            if m == "retrieval_only":
                continue
            agg = aggregate(results[m])
            delta_mrr = agg["mrr@8"] - base_agg["mrr@8"]
            delta_h1  = agg["hit@1"] - base_agg["hit@1"]
            print(f"    {m}: Δ MRR@8={delta_mrr:+.3f}  Δ Hit@1={delta_h1:+.3f}")

    rerankers = [m for m in models if m != "retrieval_only"]
    if len(rerankers) == 2:
        m1, m2 = rerankers
        agg1 = aggregate(results[m1])
        agg2 = aggregate(results[m2])
        delta_mrr = agg2["mrr@8"] - agg1["mrr@8"]
        delta_h1  = agg2["hit@1"] - agg1["hit@1"]
        print(f"\n  Δ MRR@8  ({m2} vs {m1}): {delta_mrr:+.3f}")
        print(f"  Δ Hit@1  ({m2} vs {m1}): {delta_h1:+.3f}")
        if delta_mrr > 0.02:
            print(f"\n  RECOMMENDATION: upgrade to {m2} (+{delta_mrr:.3f} MRR)")
        elif delta_mrr < -0.02:
            print(f"\n  RECOMMENDATION: keep {m1} (better by {-delta_mrr:.3f} MRR)")
        else:
            print(f"\n  RECOMMENDATION: models are comparable — prefer {m1} (faster)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark flashrank reranker models")
    parser.add_argument(
        "--models", nargs="+", default=["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"],
        metavar="MODEL",
        help=f"Models to compare. Available: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run (default: all)")
    parser.add_argument("--diverse", action="store_true",
                        help="Pick 2 questions per category across different sections (10 total)")
    parser.add_argument("--tenant", default="vanaciprime", help="Qdrant tenant_id filter")
    parser.add_argument(
        "--golden-set", default="data/golden_test_set/golden_test_set.json",
        help="Path to golden test set JSON",
    )
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Save raw results to JSON file (e.g. results/reranker_baseline.json)",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"[ERROR] Golden test set not found: {golden_path}")
        print("  Run: python scripts/golden_test_set.py")
        sys.exit(1)

    with open(golden_path) as f:
        all_questions = json.load(f)

    questions = [q for q in all_questions if q["category"] != "out_of_scope"]

    if args.diverse:
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        diverse = []
        for cat, qs in sorted(by_cat.items()):
            seen_sections: set = set()
            picked = 0
            for q in qs:
                if q["source_section"] not in seen_sections:
                    diverse.append(q)
                    seen_sections.add(q["source_section"])
                    picked += 1
                    if picked == 2:
                        break
            if picked < 2 and len(qs) > 1:
                for q in qs:
                    if q not in diverse:
                        diverse.append(q)
                        break
        questions = diverse
    elif args.limit:
        questions = questions[: args.limit]

    models_to_run = ["retrieval_only"] + args.models

    print("\nReranker Benchmark  (answer-based relevance — document agnostic)")
    print(f"  Models   : retrieval_only (RRF baseline) + {args.models}")
    print(f"  Questions: {len(questions)} (out_of_scope excluded)")
    print(f"  Tenant   : {args.tenant}")
    if args.diverse:
        print("\n  Diverse sample (2 per category, different sections):")
        for q in questions:
            print(f"    [{q['category']:12s}] [{q['source_section']:25s}] {q['question'][:55]}")
    print("\nLoading models and running queries...\n")

    results = run_benchmark(models_to_run, questions, args.tenant)
    print_report(models_to_run, results)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"models": models_to_run, "results": results}, f, indent=2)
        print(f"\n  Saved raw results → {out}")


if __name__ == "__main__":
    main()
