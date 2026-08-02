"""Before/after retrieval recall: Phase 1 baseline vs. LLM-normalized queries.

For every row in eval/labeled_ground_truth.csv this script compares two ways of
driving the RetrievalService:

  BEFORE (Phase 1 baseline)
      retrieve(raw_input) -> top-5 candidates.

  AFTER (Touchpoints 1 + 2)
      raw_input
        -> normalize_record()          (Touchpoint 1, Gemini)
        -> reformulate_queries()       (Touchpoint 2, Gemini)
        -> retrieve() each query, merge by mmv_id, keep best combined_score
        -> top-5 candidates.

Recall@5 is scored over the non-reject rows (those with an expected mmv_id);
ambiguous rows accept any of their pipe-separated ids. Reject rows are reported
separately for context. The AFTER path needs a Gemini API key
(GEMINI_API_KEY / GOOGLE_API_KEY); without one the script prints a notice and
runs the baseline only.

Usage:
    python scripts/test_retrieval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.retrieval_service import RetrievalService  # noqa: E402

GROUND_TRUTH_PATH = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"

# Recall is reported at each of these cut-offs. @5 was the agreed headline metric;
# @1/@3 are added because deterministic retrieval already saturates @5, so any
# lift from the LLM layer only shows up at tighter ranks.
RANKS = (1, 3, 5)
TOP_N = max(RANKS)


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _expected_ids(row: pd.Series) -> list[str]:
    """Parse expected_mmv_id, which may be '', 'MMVxxx', or 'A|B' (ambiguous)."""
    raw = (row.get("expected_mmv_id") or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def _top_ids(candidates: list[dict]) -> list[str]:
    return [c.get("mmv_id") for c in candidates]


def _best_rank(expected: list[str], ranked_ids: list[str]) -> int | None:
    """1-indexed rank of the highest-placed acceptable id, or None if absent."""
    best: int | None = None
    for eid in expected:
        if eid in ranked_ids:
            rank = ranked_ids.index(eid) + 1
            if best is None or rank < best:
                best = rank
    return best


def _hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


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
    ranked = sorted(
        merged.values(), key=lambda c: c["combined_score"], reverse=True
    )
    return ranked[:top_n]


def main() -> None:
    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype=str).fillna("")

    run_after = _has_api_key()
    if not run_after:
        print(
            "!! No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY).\n"
            "   Running the Phase 1 BASELINE only; the normalize+reformulate "
            "path is skipped.\n"
        )

    print("Building RetrievalService (loading model + encoding catalogue)...\n")
    service = RetrievalService()

    # Import the touchpoints lazily so the baseline-only path never needs the
    # LLM deps/key just to import this module.
    if run_after:
        from llm_touchpoints.normalization_llm import normalize_record
        from llm_touchpoints.query_reformulation_llm import reformulate_queries

    header = (
        f"{'id':<6} {'tier':<10} {'expected':<16} "
        f"{'rank_b':<7} {'rank_a':<7}  queries"
    )
    print(header)
    print("-" * len(header))

    scored = 0
    reject_total = 0
    # hits[k] = count of scored rows whose best rank <= k, per path.
    base_hits = {k: 0 for k in RANKS}
    after_hits = {k: 0 for k in RANKS}

    for _, row in gt.iterrows():
        input_id = row["input_id"]
        raw_input = row["raw_input"]
        tier = row["expected_match_type"]
        expected = _expected_ids(row)
        is_reject = not expected

        # --- BEFORE: raw query straight into retrieval ---
        before_rank = _best_rank(expected, _top_ids(service.retrieve(raw_input, top_n=TOP_N)))

        # --- AFTER: normalize -> reformulate -> multi-query retrieve -> merge ---
        after_rank = before_rank
        queries_display = "(baseline only)"
        if run_after:
            normalized = normalize_record({"raw_input": raw_input})
            queries = reformulate_queries(normalized)
            after_rank = _best_rank(expected, _top_ids(_retrieve_multi(service, queries, TOP_N)))
            queries_display = " | ".join(queries)

        if is_reject:
            reject_total += 1
            # No correct id exists for reject rows, so nothing to score for recall;
            # the reject decision itself belongs to the later verifier touchpoints.
        else:
            scored += 1
            for k in RANKS:
                base_hits[k] += int(_hit_at(before_rank, k))
                after_hits[k] += int(_hit_at(after_rank, k))

        expected_str = "|".join(expected) if expected else "(reject)"
        print(
            f"{input_id:<6} {tier:<10} {expected_str:<16} "
            f"{str(before_rank or '-'):<7} {str(after_rank or '-'):<7}  {queries_display}"
        )

    print()
    print("=" * 72)
    print(f"Rows: {len(gt)}   scored (non-reject): {scored}   reject: {reject_total}")
    print("rank_b / rank_a = best rank of the correct id, before / after ('-' = not in top-5)\n")

    def _pct(n: int, d: int) -> str:
        return f"{(100.0 * n / d):.1f}%" if d else "n/a"

    for k in RANKS:
        b = base_hits[k]
        line = f"Recall@{k}  baseline = {b}/{scored} ({_pct(b, scored)})"
        if run_after:
            a = after_hits[k]
            delta = a - b
            sign = "+" if delta >= 0 else ""
            line += (
                f"   |   after = {a}/{scored} ({_pct(a, scored)})"
                f"   Δ = {sign}{delta}"
            )
        else:
            line += "   |   after = (skipped — no API key)"
        print(line)
    print("=" * 72)


if __name__ == "__main__":
    main()
