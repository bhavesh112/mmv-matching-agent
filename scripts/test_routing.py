"""End-to-end routing test for Touchpoints 3+4 (reasoning + adversarial verifier).

Runs the real pipeline over EVERY ground-truth row, all the way through the
confidence router and the adversarial verifier, using the compiled LangGraph:

    raw_input
      -> normalize_record()      (Touchpoint 1, Gemini)
      -> reformulate_queries()   (Touchpoint 2, Gemini)
      -> retrieve() + merge      (Phase 2 retrieval -> candidate_records)
      -> build_graph().invoke()  (Touchpoint 3 reasoning -> confidence router
                                  -> Touchpoint 4 verifier on the >0.95 tier
                                  -> approve / review / reject)

For each row it prints the route taken (and the verifier's verdict + concern
when the verifier fired), then prints a ROUTING SUMMARY: how many records were
auto-approved, downgraded to review by the verifier, sent straight to review
(the 0.80-0.95 tier), and rejected. Needs a Gemini API key (GEMINI_API_KEY /
GOOGLE_API_KEY); without one it prints a notice and exits.

Usage:
    python scripts/test_routing.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env up front so the API-key check sees GEMINI_API_KEY before any
# touchpoint module is imported.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it.
    pass

import config  # noqa: E402
from services.retrieval_service import RetrievalService  # noqa: E402

GROUND_TRUTH_PATH = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"

# Candidate-set size handed to the reasoning agent per record.
CANDIDATE_TOP_N = 8


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _expected_ids(row: pd.Series) -> list[str]:
    raw = (row.get("expected_mmv_id") or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def _retrieve_multi(
    service: RetrievalService, queries: list[str], top_n: int
) -> list[dict]:
    """Run each query, merge candidates by mmv_id keeping the best combined_score."""
    merged: dict[str, dict] = {}
    for query in queries:
        for cand in service.retrieve(query, top_n=top_n):
            mmv_id = cand.get("mmv_id")
            existing = merged.get(mmv_id)
            if existing is None or cand["combined_score"] > existing["combined_score"]:
                merged[mmv_id] = cand
    ranked = sorted(merged.values(), key=lambda c: c["combined_score"], reverse=True)
    return ranked[:top_n]


def _classify(final_state: dict) -> str:
    """Map a finished graph state to one of the four routing buckets."""
    decision = final_state.get("final_decision")
    verifier = final_state.get("verifier_result") or {}
    verifier_ran = bool(verifier)  # only the >0.95 tier populates verifier_result

    if decision == "approved":
        return "auto_approved"
    if decision == "review":
        return "downgraded_by_verifier" if verifier_ran else "sent_to_review"
    if decision == "rejected":
        return "rejected"
    return "unknown"


_BUCKET_LABEL = {
    "auto_approved": "AUTO-APPROVED",
    "downgraded_by_verifier": "REVIEW (downgraded by verifier)",
    "sent_to_review": "REVIEW (0.80-0.95 tier)",
    "rejected": "REJECTED",
    "unknown": "UNKNOWN",
}


def main() -> None:
    if not _has_api_key():
        print(
            "!! No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY).\n"
            "   The reasoning + verifier touchpoints need the LLM; nothing to run."
        )
        return

    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype=str).fillna("")

    print("Building RetrievalService (loading model + encoding catalogue)...\n")
    service = RetrievalService()

    from graph import build_graph
    from llm_touchpoints.normalization_llm import normalize_record
    from llm_touchpoints.query_reformulation_llm import reformulate_queries

    app = build_graph()

    print(
        f"Routing tiers: confidence > {config.AUTO_APPROVE_CONFIDENCE_THRESHOLD} "
        f"-> verifier -> approve/review;  "
        f"{config.REVIEW_CONFIDENCE_FLOOR} <= confidence <= "
        f"{config.AUTO_APPROVE_CONFIDENCE_THRESHOLD} -> review;  "
        f"< {config.REVIEW_CONFIDENCE_FLOOR} -> reject.\n"
    )

    counts: dict[str, int] = {
        "auto_approved": 0,
        "downgraded_by_verifier": 0,
        "sent_to_review": 0,
        "rejected": 0,
        "unknown": 0,
    }
    rows_out: list[dict] = []

    for _, row in gt.iterrows():
        input_id = row["input_id"]
        raw_input = row["raw_input"]
        tier = row["expected_match_type"]
        expected = _expected_ids(row)
        expected_str = "|".join(expected) if expected else "(reject)"

        normalized = normalize_record({"raw_input": raw_input})
        queries = reformulate_queries(normalized)
        candidates = _retrieve_multi(service, queries, CANDIDATE_TOP_N)

        final_state = app.invoke(
            {
                "input_record": {"raw_input": raw_input},
                "normalized_record": normalized,
                "candidate_records": candidates,
            }
        )

        bucket = _classify(final_state)
        counts[bucket] += 1

        match = final_state.get("selected_match") or {}
        chosen_id = match.get("mmv_id")
        confidence = final_state.get("confidence")
        verifier = final_state.get("verifier_result") or {}

        rows_out.append(
            {
                "input_id": input_id,
                "tier": tier,
                "expected": expected_str,
                "chosen": chosen_id,
                "confidence": confidence,
                "bucket": bucket,
                "verdict": verifier.get("verdict", ""),
                "concern": verifier.get("concern", ""),
            }
        )

        print("=" * 78)
        print(f"{input_id} [{tier}]  raw_input: {raw_input!r}")
        print(
            f"  match={chosen_id}  confidence={confidence}  "
            f"expected={expected_str}"
        )
        if verifier:
            print(
                f"  verifier: {verifier.get('verdict', '?').upper()}"
                + (f"  concern: {verifier.get('concern')}" if verifier.get("concern") else "")
            )
        print(f"  -> {_BUCKET_LABEL[bucket]}")

    total = sum(counts.values())
    print("\n" + "#" * 78)
    print(f"ROUTING SUMMARY over {total} ground-truth records")
    print("#" * 78)
    print(f"  auto-approved (>0.95, verifier passed) : {counts['auto_approved']}")
    print(f"  downgraded to review by verifier       : {counts['downgraded_by_verifier']}")
    print(f"  sent to review (0.80-0.95 tier)        : {counts['sent_to_review']}")
    print(f"  rejected (<0.80)                       : {counts['rejected']}")
    if counts["unknown"]:
        print(f"  unknown (no final_decision)            : {counts['unknown']}")
    total_review = counts["downgraded_by_verifier"] + counts["sent_to_review"]
    print("-" * 78)
    print(f"  total sent to manual review            : {total_review}")
    print("#" * 78)


if __name__ == "__main__":
    main()
