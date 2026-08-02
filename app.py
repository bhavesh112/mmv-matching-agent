"""Application entrypoint for the MMV matching pipeline (CLI).

Drives the FULL pipeline end to end over the labeled ground-truth set:

    raw_input
      -> normalize_record()      (Touchpoint 1, Gemini)
      -> reformulate_queries()   (Touchpoint 2, Gemini)
      -> retrieve() + merge      (Phase 2 retrieval -> candidate_records)
      -> build_graph().invoke()  (Touchpoints 3-5 + audit):
             reason -> route -> [verify] -> approve/review/reject
                    -> explain -> audit -> END

The pre-graph steps (normalize / reformulate / retrieve) run here in the driver
because they need the heavy RetrievalService; the graph itself starts at the
reasoning node, which consumes ``candidate_records`` (see graph.py).

Every record is guaranteed to produce a ``final_decision``, an ``explanation``,
and a complete audit-log entry — even one whose LLM calls fail on quota: such a
record is failed safe to "review" with a deterministic explanation and logged
explicitly so the audit trail stays complete.

Usage:
    python app.py                    # all ground-truth records
    python app.py --ids GT01,GT11    # just these input_ids
    python app.py --limit 5          # first 5 records
    python app.py --keep-log         # append to the existing audit log

Needs a Gemini API key (GEMINI_API_KEY / GOOGLE_API_KEY).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing any touchpoint so the API key is visible up front.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it.
    pass

import os  # noqa: E402

import config  # noqa: E402
from graph import build_graph  # noqa: E402
from llm_touchpoints.explanation_llm import explain_decision  # noqa: E402
from services.audit_logger import AuditLogger  # noqa: E402

GROUND_TRUTH_PATH = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"

# Candidate-set size handed to the reasoning agent per record.
CANDIDATE_TOP_N = 8


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _retrieve_multi(service, queries: list[str], top_n: int) -> list[dict]:
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


def _fallback_state(input_record: dict, exc: Exception) -> dict:
    """Build a complete, failed-safe state for a record whose pre-graph LLM steps
    threw (almost always a Gemini quota / 429). Fails to "review" with a
    deterministic explanation so the audit trail is never missing a record."""
    normalized = {"raw_input": input_record.get("raw_input")}
    explanation = explain_decision(
        final_decision="review",
        normalized_record=normalized,
        candidate_records=[],
        selected_match=None,
        validation_status="not_validated",
        confidence=0.0,
        reason=f"pipeline error before a decision could be made: {exc}",
        verifier_result=None,
        # Use the deterministic fallback path only — do not spend an LLM call to
        # explain a record we already failed on quota.
        service=_NullExplanationService(),
    )["explanation"]
    return {
        "input_record": input_record,
        "normalized_record": normalized,
        "search_queries": [],
        "candidate_records": [],
        "selected_match": None,
        "reasoning_decision": {},
        "reasoning_trace": [],
        "validation_status": "not_validated",
        "verifier_result": None,
        "confidence": 0.0,
        "explanation": explanation,
        "final_decision": "review",
        "step_count": 0,
        "llm_call_log": [],
        "error": str(exc),
    }


class _NullExplanationService:
    """Stand-in that forces explain_decision onto its deterministic fallback
    (no LLM call) — used when we already know the record failed on quota."""

    call_log: list = []

    def generate_json(self, *args, **kwargs):  # noqa: D401 - see class docstring
        raise RuntimeError("explanation LLM intentionally skipped for failed record")


def _load_rows(ids: list[str] | None, limit: int | None) -> pd.DataFrame:
    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype=str).fillna("")
    if ids:
        available = set(gt["input_id"])
        ordered = [i for i in ids if i in available]  # preserve requested order
        gt = gt.set_index("input_id").loc[ordered].reset_index()
    if limit is not None:
        gt = gt.head(limit)
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MMV matching pipeline.")
    parser.add_argument("--ids", default="", help="comma-separated input_ids (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument(
        "--keep-log",
        action="store_true",
        help="append to the existing audit log instead of starting fresh",
    )
    args = parser.parse_args()

    if not _has_api_key():
        print(
            "!! No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY).\n"
            "   The pipeline needs the LLM touchpoints. Set a key and retry."
        )
        raise SystemExit(1)

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
    rows = _load_rows(ids, args.limit)
    if rows.empty:
        print("No matching ground-truth rows to process.")
        raise SystemExit(1)

    # Fresh audit log per run by default so read-back reflects THIS run only.
    logger = AuditLogger(reset=not args.keep_log)
    print(f"Audit log: {logger.path}")
    print("Building RetrievalService (loading model + encoding catalogue)...\n")

    from services.retrieval_service import RetrievalService
    from llm_touchpoints.normalization_llm import normalize_record
    from llm_touchpoints.query_reformulation_llm import reformulate_queries

    retrieval = RetrievalService()
    app = build_graph(audit_logger=logger)

    processed_ids: list[str] = []
    decisions: dict[str, int] = {"approved": 0, "review": 0, "rejected": 0}

    for _, row in rows.iterrows():
        input_id = row["input_id"]
        raw_input = row["raw_input"]
        tier = row["expected_match_type"]
        input_record = {"input_id": input_id, "raw_input": raw_input}

        print("=" * 78)
        print(f"{input_id} [{tier}]  raw_input: {raw_input!r}")
        print("-" * 78)

        try:
            normalized = normalize_record({"raw_input": raw_input})
            queries = reformulate_queries(normalized)
            candidates = _retrieve_multi(retrieval, queries, CANDIDATE_TOP_N)

            final_state = app.invoke(
                {
                    "input_record": input_record,
                    "normalized_record": normalized,
                    "candidate_records": candidates,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep the run going, log the record
            # Pre-graph LLM step failed (usually quota). Fail safe to review and
            # log the record explicitly so the audit trail stays complete.
            final_state = _fallback_state(input_record, exc)
            logger.log(final_state)
            print(f"!! pipeline error (failed safe to review): {exc}")

        decision = final_state.get("final_decision", "review")
        explanation = final_state.get("explanation", "")
        decisions[decision] = decisions.get(decision, 0) + 1
        processed_ids.append(input_id)

        match = final_state.get("selected_match") or {}
        print(
            f"DECISION: {decision}  "
            f"match={match.get('mmv_id')}  "
            f"confidence={final_state.get('confidence')}  "
            f"validation={final_state.get('validation_status')}"
        )
        print(f"EXPLANATION: {explanation}")
        print()

    # --- Confirm every record produced a complete audit entry -------------
    entries = logger.read_all()
    by_record = {e.get("record_id"): e for e in entries}

    print("=" * 78)
    print("RUN SUMMARY")
    print("-" * 78)
    print(f"records processed : {len(processed_ids)}")
    print(f"audit entries     : {len(entries)}")
    print(
        "decisions         : "
        + ", ".join(f"{k}={v}" for k, v in decisions.items() if v)
    )

    complete = 0
    incomplete: list[str] = []
    for input_id in processed_ids:
        entry = by_record.get(input_id)
        ok = bool(
            entry
            and entry.get("final_decision")
            and (entry.get("explanation") or "").strip()
        )
        if ok:
            complete += 1
        else:
            incomplete.append(input_id)

    print(
        f"complete entries  : {complete}/{len(processed_ids)} "
        "(final_decision + explanation + audit entry)"
    )
    if incomplete:
        print(f"INCOMPLETE        : {', '.join(incomplete)}")
        print("=" * 78)
        raise SystemExit(1)

    print("OK: every record produced a final_decision, an explanation, and a "
          "complete audit entry.")
    print("=" * 78)


if __name__ == "__main__":
    main()
