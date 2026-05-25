"""
Offline Document Quality Evaluator
────────────────────────────────────
Runs sample questions through the pipeline and measures answer quality.
Stores results in Neon via SQLAlchemy. No raw SQL queries.
 
Usage:
    uv run python scripts/eval_document.py --document_id abc123
    uv run python scripts/eval_document.py --recent
    uv run python scripts/eval_document.py --recent --hours 48
"""
 
import argparse
import json
import os
import uuid
from datetime import datetime, timedelta
 
from fastapi.testclient import TestClient
from sqlalchemy import desc
 
from api.main import app
from rag.db import SessionLocal, Document, DocumentEvalResult
 
client = TestClient(app)
PASS_THRESHOLD = 0.80
 
 
def evaluate_document(document_id: str) -> dict:
    """
    Run sample questions for a document through the pipeline.
    Stores pass/fail results in Neon document_eval_results table.
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(
            Document.document_id == document_id
        ).first()
 
        if not doc:
            print(f"[EVAL] Document not found: {document_id}")
            return None
 
        if not doc.sample_questions or not doc.sample_questions.strip():
            print(f"[EVAL] No sample questions for {doc.filename} — skipping")
            return None
 
        questions = [
            q.strip()
            for q in doc.sample_questions.strip().split("\n")
            if q.strip()
        ]
 
        print(f"[EVAL] Evaluating {doc.filename} with {len(questions)} questions")
 
        details = []
        passed = 0
 
        for question in questions:
            response = client.post(
                "/api/v1/chat",
                json={"question": question, "thread_id": f"eval_{document_id}"},
            )
 
            if response.status_code != 200:
                details.append({
                    "question": question,
                    "answer": f"Error: HTTP {response.status_code}",
                    "passed": False,
                    "reason": "api_error",
                })
                continue
 
            answer = response.json().get("answer", "")
            answer_lower = answer.lower()
 
            refusal_phrases = [
                "i don't have enough information",
                "context does not",
                "cannot find",
                "not available in",
                "not mentioned in",
            ]
            is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
            is_substantive = len(answer) > 50
 
            test_passed = is_substantive and not is_refusal
            if test_passed:
                passed += 1
 
            details.append({
                "question": question,
                "answer": answer[:200],
                "passed": test_passed,
                "reason": "refusal" if is_refusal else (
                    "too_short" if not is_substantive else "ok"
                ),
            })
 
            result_symbol = "PASS" if test_passed else "FAIL"
            print(f"  [{result_symbol}] {question[:60]}")
 
        pass_rate = passed / len(questions) if questions else 0
        if pass_rate >= PASS_THRESHOLD:
            eval_status = "passing"
        elif pass_rate >= 0.5:
            eval_status = "warning"
        else:
            eval_status = "failing"
 
        eval_result = DocumentEvalResult(
            id=str(uuid.uuid4()),
            document_id=document_id,
            questions_tested=len(questions),
            questions_passed=passed,
            pass_rate=pass_rate,
            threshold=PASS_THRESHOLD,
            status=eval_status,
        )
        db.add(eval_result)
        db.commit()
 
        print(
            f"[EVAL] {doc.filename}: {passed}/{len(questions)} passed "
            f"({pass_rate:.0%}) — {eval_status.upper()}"
        )
 
        return {
            "document_id": document_id,
            "filename": doc.filename,
            "pass_rate": pass_rate,
            "status": eval_status,
            "details": details,
        }
 
    except Exception as e:
        db.rollback()
        print(f"[EVAL] Error evaluating {document_id}: {e}")
        return None
    finally:
        db.close()
 
 
def evaluate_recent(hours: int = 24) -> list:
    """Evaluate all documents ingested in the last N hours."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        documents = (
            db.query(Document)
            .filter(
                Document.upload_date > cutoff,
                Document.status == "complete",
            )
            .all()
        )
    finally:
        db.close()
 
    print(f"[EVAL] Found {len(documents)} documents to evaluate")
    results = []
 
    for doc in documents:
        result = evaluate_document(doc.document_id)
        if result:
            results.append(result)
 
    failing = [r for r in results if r["status"] == "failing"]
    if failing:
        print(f"\nWARNING: {len(failing)} documents below threshold:")
        for r in failing:
            print(f"  {r['filename']}: {r['pass_rate']:.0%}")
    else:
        print(f"\nAll {len(results)} documents passed quality checks.")
 
    return results
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline document quality evaluator")
    parser.add_argument("--document_id", help="Evaluate a specific document by ID")
    parser.add_argument(
        "--recent",
        action="store_true",
        help="Evaluate all documents from recent hours",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours back to look for recent documents (default: 24)",
    )
    args = parser.parse_args()
 
    if args.document_id:
        evaluate_document(args.document_id)
    elif args.recent:
        evaluate_recent(hours=args.hours)
    else:
        parser.print_help()
 