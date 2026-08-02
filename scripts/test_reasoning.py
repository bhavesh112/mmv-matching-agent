"""End-to-end smoke test for Touchpoint 3 (ReAct reasoning agent).

Runs the real pipeline over a curated set of ground-truth rows spanning all
three tiers (match / ambiguous / reject):

    raw_input
      -> normalize_record()      (Touchpoint 1, Gemini)
      -> reformulate_queries()   (Touchpoint 2, Gemini)
      -> retrieve() + merge      (Phase 2 retrieval -> candidate_records)
      -> run_reasoning_agent()   (Touchpoint 3, ReAct agent + tools)

For each row it prints the full ReAct trace (thought / action / observation per
step) and the final {match, confidence, reason}, then checks the chosen match
against the expected id(s). Needs a Gemini API key (GEMINI_API_KEY /
GOOGLE_API_KEY); without one it prints a notice and exits.

Usage:
    python scripts/test_reasoning.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env up front so the API-key check below sees GEMINI_API_KEY before any
# touchpoint module is imported (llm_service also loads it, but only lazily).
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it.
    pass

from services.retrieval_service import RetrievalService  # noqa: E402

GROUND_TRUTH_PATH = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"

# Curated subset spanning all three tiers: 1 clean match, 2 ambiguous, 2 rejects
# (5 rows). Kept small on purpose — the Gemini free tier allows ~20 requests/day
# and each row costs ~4 LLM calls (normalize + reformulate + a few agent steps),
# so a full run stays within a single day's free-tier budget. Widen this list if
# you have paid quota. Override with the MMV_TEST_SAMPLE env var (comma-separated
# input_ids).
SAMPLE_IDS = tuple(
    s.strip()
    for s in os.environ.get(
        "MMV_TEST_SAMPLE", "GT01,GT11,GT13,GT10,GT16"
    ).split(",")
    if s.strip()
)

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


def _summarize_observation(obs: object) -> str:
    """Compact one-line view of a tool observation for the printed trace."""
    if obs is None:
        return "(none)"
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        if "error" in obs:
            return f"error: {obs['error']}"
        if "validation_status" in obs:  # validate()
            viol = obs.get("violations") or []
            detail = "; ".join(v.get("detail", "") for v in viol)
            return f"validation_status={obs['validation_status']}" + (
                f" | violations: {detail}" if detail else ""
            )
        if "summary" in obs:  # attribute_compare()
            s = obs["summary"]
            return (
                f"mismatches={s.get('mismatches')} "
                f"input_missing={s.get('input_missing')}"
            )
        if "total_candidates" in obs:  # broaden_candidates()
            return (
                f"broadened: +{obs.get('added')} -> "
                f"{obs.get('total_candidates')} candidates"
            )
    return json.dumps(obs, ensure_ascii=False)[:200]


def _print_trace(trace: list[dict]) -> None:
    for entry in trace:
        print(f"  step {entry.get('step')}: [{entry.get('action')}]")
        thought = (entry.get("thought") or "").strip()
        if thought:
            print(f"    thought: {thought}")
        ai = entry.get("action_input")
        if ai:
            print(f"    action_input: {json.dumps(ai, ensure_ascii=False)}")
        print(f"    observation: {_summarize_observation(entry.get('observation'))}")


def _verdict(expected: list[str], chosen_id: str | None, tier: str) -> str:
    is_reject = not expected
    if is_reject:
        return "PASS" if chosen_id is None else "FAIL"
    if chosen_id is None:
        return "FAIL"
    return "PASS" if chosen_id in expected else "FAIL"


def main() -> None:
    if not _has_api_key():
        print(
            "!! No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY).\n"
            "   The reasoning agent needs the LLM; nothing to run. Set a key and "
            "retry."
        )
        return

    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype=str).fillna("")
    rows = gt[gt["input_id"].isin(SAMPLE_IDS)]
    # Preserve the curated order.
    rows = rows.set_index("input_id").loc[list(SAMPLE_IDS)].reset_index()

    print("Building RetrievalService (loading model + encoding catalogue)...\n")
    service = RetrievalService()

    from llm_touchpoints.normalization_llm import normalize_record
    from llm_touchpoints.query_reformulation_llm import reformulate_queries
    from llm_touchpoints.reasoning_llm import run_reasoning_agent

    def retrieve_fn(query: str, top_n: int) -> list:
        return service.retrieve(query, top_n=top_n)

    passes = 0
    total = 0
    skipped = 0
    for _, row in rows.iterrows():
        input_id = row["input_id"]
        raw_input = row["raw_input"]
        tier = row["expected_match_type"]
        expected = _expected_ids(row)
        expected_str = "|".join(expected) if expected else "(reject)"

        print("=" * 78)
        print(f"{input_id} [{tier}]  raw_input: {raw_input!r}")
        print(f"expected: {expected_str}")
        print("-" * 78)

        try:
            normalized = normalize_record({"raw_input": raw_input})
            queries = reformulate_queries(normalized)
            candidates = _retrieve_multi(service, queries, CANDIDATE_TOP_N)

            print(f"queries: {queries}")
            print(
                "candidates: "
                + ", ".join(
                    f"{c.get('mmv_id')}({c.get('combined_score')})"
                    for c in candidates
                )
            )
            print("reasoning trace:")

            outcome = run_reasoning_agent(
                normalized, candidates, retrieve_fn=retrieve_fn
            )
        except RuntimeError as exc:
            # Almost always a Gemini quota / rate-limit (429) on the free tier.
            # Skip this record rather than aborting the whole run so already-
            # completed records still report.
            skipped += 1
            print(f"SKIPPED (LLM error): {exc}")
            print()
            continue

        _print_trace(outcome["reasoning_trace"])

        match = outcome["match"]
        chosen_id = match.get("mmv_id") if match else None
        verdict = _verdict(expected, chosen_id, tier)
        total += 1
        passes += int(verdict == "PASS")

        print("-" * 78)
        print(
            f"DECISION: match={chosen_id} "
            f"confidence={outcome['confidence']} "
            f"validation_status={outcome['validation_status']} "
            f"steps={outcome['step_count']}"
        )
        print(f"reason: {outcome['reason']}")
        print(f"VERDICT: {verdict}  (expected {expected_str})")
        print()

    print("=" * 78)
    print(f"Passed {passes}/{total} scored records.", end="")
    if skipped:
        print(f"  ({skipped} skipped on LLM/quota errors)", end="")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
