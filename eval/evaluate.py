"""Evaluation harness for the MMV matching pipeline.

Runs the FULL pipeline (normalize -> reformulate -> retrieve -> reason -> route
-> [verify] -> approve/review/reject -> explain -> audit) over
``eval/labeled_ground_truth.csv`` and reports:

  * **Accuracy vs ground truth** — does the pipeline's decision tier (and, for
    matches, the chosen mmv_id) agree with the label? Plus a confusion matrix.
  * **Precision at the auto-approve tier, WITH vs WITHOUT Touchpoint 4** — of the
    records the pipeline auto-approves, what fraction are genuinely correct?
    Because the adversarial verifier (T4) can only ever *downgrade* an
    auto-approval to review (never promote a record — see ``graph.build_graph``),
    the no-verifier counterfactual is exact and needs no second run: the
    "without T4" approved set is every record that reached the verifier tier
    (reasoning confidence > ``AUTO_APPROVE_CONFIDENCE_THRESHOLD``), and the
    "with T4" set is the subset the verifier passed. So we run the pipeline once
    and derive both precisions.
  * **Per-touchpoint cost and latency** — aggregated from each record's
    ``llm_call_log`` (the normalization/reformulation calls, which run in this
    driver rather than inside the graph, are merged back in so the accounting is
    complete). Cost is token counts x approximate published Gemini rates.
  * **% of auto-approve candidates downgraded by the verifier** — of the records
    that reached the verifier tier, the fraction the verifier failed.
  * **ReAct step-count distribution** — how many reasoning steps records took.

Prints a summary table and writes one row per record to ``eval/results.csv``.

Needs a Gemini API key (GEMINI_API_KEY / GOOGLE_API_KEY). Records whose pre-graph
LLM steps fail (typically a quota 429) are failed safe to "review" exactly as the
app driver does, so the run always completes and every record is scored.

Usage:
    python -m eval.evaluate                 # full ground-truth set
    python -m eval.evaluate --ids GT01,GT11 # just these input_ids
    python -m eval.evaluate --limit 5       # first 5 records
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing any touchpoint so the API key is visible up front.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it.
    pass

import config  # noqa: E402
from app import (  # noqa: E402  (reuse the real driver's helpers verbatim)
    CANDIDATE_TOP_N,
    _fallback_state,
    _has_api_key,
    _load_rows,
    _retrieve_multi,
)
from graph import build_graph  # noqa: E402
from services.audit_logger import AuditLogger  # noqa: E402
from services.llm_service import LLMService  # noqa: E402

GROUND_TRUTH_PATH = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"
RESULTS_PATH = _PROJECT_ROOT / "eval" / "results.csv"
# Keep the eval audit trail out of the app's own log file.
EVAL_AUDIT_LOG_PATH = "logs/eval_audit_log.jsonl"

# Which ground-truth tier maps to which pipeline decision.
_EXPECTED_DECISION = {"match": "approved", "ambiguous": "review", "reject": "rejected"}

# Touchpoints reported in the cost/latency table, in pipeline order.
_TOUCHPOINT_ORDER = (
    "normalization",
    "query_reformulation",
    "reasoning",
    "verifier",
    "explanation",
)

# Approximate published Gemini API prices (USD per 1M tokens). Used only to turn
# the token counts captured in llm_call_log (from usage_metadata) into a rough
# cost estimate; adjust here if Google's rates change. flash-lite is the cheap
# tier the explanation touchpoint runs on (see config.EXPLANATION_MODEL).
_PRICING = {
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
}
_DEFAULT_PRICING = {"input": 0.30, "output": 2.50}


# --- small helpers --------------------------------------------------------


def _parse_expected_ids(raw: object) -> list[str]:
    """Ground-truth ids are '|'-separated for ambiguous rows, empty for reject."""
    return [s.strip() for s in str(raw or "").split("|") if s.strip()]


def _call_cost(call: dict) -> float:
    """USD cost of a single llm_call_log entry from its token counts."""
    rate = _PRICING.get(call.get("model"), _DEFAULT_PRICING)
    prompt = call.get("prompt_tokens") or 0
    completion = call.get("completion_tokens") or 0
    return prompt / 1e6 * rate["input"] + completion / 1e6 * rate["output"]


def _fmt_usd(value: float) -> str:
    return f"${value:.5f}"


# --- per-record pipeline run ----------------------------------------------


def _run_record(app, retrieval, input_record: dict) -> dict:
    """Run the full pipeline for one record and return its final MMVState.

    Normalization and reformulation run here (they need the retrieval service),
    so we give them dedicated LLMService instances and splice their call logs
    onto the front of the graph's ``llm_call_log`` — otherwise the per-touchpoint
    cost/latency accounting would miss the two pre-graph touchpoints.
    """
    from llm_touchpoints.normalization_llm import normalize_record
    from llm_touchpoints.query_reformulation_llm import reformulate_queries

    norm_service = LLMService()
    reform_service = LLMService()

    def _pre_graph_calls() -> list:
        return list(norm_service.call_log) + list(reform_service.call_log)

    try:
        normalized = normalize_record(
            {"raw_input": input_record["raw_input"]}, service=norm_service
        )
        queries = reformulate_queries(normalized, service=reform_service)
        candidates = _retrieve_multi(retrieval, queries, CANDIDATE_TOP_N)

        final_state = app.invoke(
            {
                "input_record": input_record,
                "normalized_record": normalized,
                "candidate_records": candidates,
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep the eval going; score the record
        final_state = _fallback_state(input_record, exc)

    # Prepend the pre-graph touchpoint calls so llm_call_log covers all five.
    final_state["llm_call_log"] = _pre_graph_calls() + (
        final_state.get("llm_call_log") or []
    )
    return final_state


def _score_record(row: dict, state: dict) -> dict:
    """Compare one final state against its ground-truth row -> a results row."""
    expected_tier = row["expected_match_type"]
    expected_ids = _parse_expected_ids(row["expected_mmv_id"])

    decision = state.get("final_decision", "review")
    selected = state.get("selected_match") or {}
    predicted_id = selected.get("mmv_id")
    confidence = float(state.get("confidence") or 0.0)

    verifier_result = state.get("verifier_result")
    reached_verifier = verifier_result is not None
    verifier_passed = bool(verifier_result and verifier_result.get("passed"))
    downgraded = reached_verifier and not verifier_passed

    decision_ok = decision == _EXPECTED_DECISION.get(expected_tier)
    if expected_tier == "reject":
        id_ok = predicted_id is None
    else:  # match / ambiguous: the chosen id must be one of the acceptable ids
        id_ok = predicted_id in expected_ids
    correct = decision_ok and id_ok

    # Per-touchpoint cost/latency for this record (for the results.csv columns).
    cost_by_tp: dict[str, float] = defaultdict(float)
    latency_by_tp: dict[str, float] = defaultdict(float)
    for call in state.get("llm_call_log") or []:
        tp = call.get("touchpoint", "unknown")
        cost_by_tp[tp] += _call_cost(call)
        latency_by_tp[tp] += call.get("latency_s") or 0.0

    return {
        "input_id": row["input_id"],
        "raw_input": row["raw_input"],
        "expected_match_type": expected_tier,
        "expected_mmv_id": row["expected_mmv_id"],
        "predicted_decision": decision,
        "predicted_mmv_id": predicted_id or "",
        "confidence": round(confidence, 4),
        "reached_verifier": reached_verifier,
        "verifier_passed": verifier_passed,
        "downgraded_by_verifier": downgraded,
        "step_count": int(state.get("step_count") or 0),
        "decision_correct": decision_ok,
        "id_correct": id_ok,
        "correct": correct,
        "total_cost_usd": round(sum(cost_by_tp.values()), 6),
        "total_latency_s": round(sum(latency_by_tp.values()), 4),
        "reasoning_latency_s": round(latency_by_tp.get("reasoning", 0.0), 4),
        "verifier_latency_s": round(latency_by_tp.get("verifier", 0.0), 4),
        "explanation_latency_s": round(latency_by_tp.get("explanation", 0.0), 4),
        # Retain the raw call log so aggregate reporting doesn't re-derive it.
        "_llm_call_log": state.get("llm_call_log") or [],
    }


# --- reporting ------------------------------------------------------------


def _print_accuracy(rows: list[dict]) -> None:
    n = len(rows)
    strict = sum(r["correct"] for r in rows)
    tier = sum(r["decision_correct"] for r in rows)
    print("ACCURACY VS GROUND TRUTH")
    print("-" * 60)
    print(f"  strict accuracy (tier + id) : {strict}/{n}  ({strict / n:.1%})")
    print(f"  tier accuracy   (decision)  : {tier}/{n}  ({tier / n:.1%})")

    # Per-expected-tier breakdown.
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["expected_match_type"]].append(r)
    print("\n  by expected tier (strict correct):")
    for tier_name in ("match", "ambiguous", "reject"):
        group = by_tier.get(tier_name, [])
        if not group:
            continue
        ok = sum(r["correct"] for r in group)
        print(f"    {tier_name:<10}: {ok}/{len(group)}")

    # Confusion matrix: expected tier x predicted decision.
    decisions = ("approved", "review", "rejected")
    print("\n  confusion (rows=expected tier, cols=predicted decision):")
    header = "    " + f"{'expected\\pred':<14}" + "".join(f"{d:>10}" for d in decisions)
    print(header)
    for tier_name in ("match", "ambiguous", "reject"):
        group = by_tier.get(tier_name, [])
        counts = Counter(r["predicted_decision"] for r in group)
        line = f"    {tier_name:<14}" + "".join(f"{counts.get(d, 0):>10}" for d in decisions)
        print(line)
    print()


def _print_auto_approve_precision(rows: list[dict]) -> None:
    # "Without T4": every record that reached the verifier tier would auto-approve.
    without_t4 = [r for r in rows if r["reached_verifier"]]
    # "With T4": the subset the verifier passed == pipeline's actual approvals.
    with_t4 = [r for r in rows if r["predicted_decision"] == "approved"]

    def _precision(approved: list[dict]) -> tuple[int, int]:
        # An auto-approval is correct iff the label is a 'match' with the right id.
        correct = sum(
            r["expected_match_type"] == "match" and r["id_correct"] for r in approved
        )
        return correct, len(approved)

    cw, nw = _precision(without_t4)
    ca, na = _precision(with_t4)

    print("AUTO-APPROVE PRECISION (Touchpoint 4 A/B)")
    print("-" * 60)
    print(
        f"  without verifier : {cw}/{nw} correct"
        + (f"  ({cw / nw:.1%})" if nw else "  (no auto-approvals)")
    )
    print(
        f"  with verifier    : {ca}/{na} correct"
        + (f"  ({ca / na:.1%})" if na else "  (no auto-approvals)")
    )
    delta = (ca / na if na else 0.0) - (cw / nw if nw else 0.0)
    print(f"  precision delta from T4 : {delta:+.1%}")

    # Verifier downgrade rate: of records that reached T4, how many it failed.
    reached = without_t4  # same set
    downgraded = [r for r in reached if r["downgraded_by_verifier"]]
    if reached:
        print(
            f"  auto-approve candidates downgraded by verifier : "
            f"{len(downgraded)}/{len(reached)}  ({len(downgraded) / len(reached):.1%})"
        )
    else:
        print("  auto-approve candidates downgraded by verifier : 0/0 (none reached T4)")
    print()


def _print_cost_latency(rows: list[dict]) -> None:
    agg: dict[str, dict] = {
        tp: {"calls": 0, "ok": 0, "failed": 0, "latency_s": 0.0, "cost": 0.0}
        for tp in _TOUCHPOINT_ORDER
    }
    for r in rows:
        for call in r["_llm_call_log"]:
            tp = call.get("touchpoint", "unknown")
            bucket = agg.setdefault(
                tp, {"calls": 0, "ok": 0, "failed": 0, "latency_s": 0.0, "cost": 0.0}
            )
            bucket["calls"] += 1
            bucket["ok" if call.get("ok") else "failed"] += 1
            bucket["latency_s"] += call.get("latency_s") or 0.0
            bucket["cost"] += _call_cost(call)

    n = len(rows) or 1
    print("PER-TOUCHPOINT COST & LATENCY")
    print("-" * 78)
    print(
        f"  {'touchpoint':<20}{'calls':>7}{'fail':>6}"
        f"{'latency_s':>12}{'lat/rec':>10}{'cost_usd':>12}"
    )
    total_calls = total_fail = 0
    total_latency = total_cost = 0.0
    # Report known touchpoints in pipeline order, then any unexpected extras.
    ordered = list(_TOUCHPOINT_ORDER) + [
        tp for tp in agg if tp not in _TOUCHPOINT_ORDER
    ]
    for tp in ordered:
        b = agg.get(tp)
        if not b or b["calls"] == 0:
            continue
        print(
            f"  {tp:<20}{b['calls']:>7}{b['failed']:>6}"
            f"{b['latency_s']:>12.3f}{b['latency_s'] / n:>10.3f}"
            f"{_fmt_usd(b['cost']):>12}"
        )
        total_calls += b["calls"]
        total_fail += b["failed"]
        total_latency += b["latency_s"]
        total_cost += b["cost"]
    print("  " + "-" * 65)
    print(
        f"  {'TOTAL':<20}{total_calls:>7}{total_fail:>6}"
        f"{total_latency:>12.3f}{total_latency / n:>10.3f}{_fmt_usd(total_cost):>12}"
    )
    print(
        f"\n  per-record average : {_fmt_usd(total_cost / n)} cost, "
        f"{total_latency / n:.3f}s latency  (over {len(rows)} records)"
    )
    print()


def _print_step_distribution(rows: list[dict]) -> None:
    counts = Counter(r["step_count"] for r in rows)
    steps = [r["step_count"] for r in rows]
    print("REACT STEP-COUNT DISTRIBUTION")
    print("-" * 60)
    if steps:
        print(
            f"  min={min(steps)}  max={max(steps)}  "
            f"mean={sum(steps) / len(steps):.2f}"
        )
    max_bar = max(counts.values()) if counts else 0
    for step in sorted(counts):
        bar = "#" * counts[step]
        print(f"  {step:>2} steps : {counts[step]:>3}  {bar}" if max_bar else "")
    print()


def _write_results_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")] if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


# --- entrypoint -----------------------------------------------------------


def evaluate(
    ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
    results_path: Path = RESULTS_PATH,
) -> dict:
    """Run the pipeline over the labeled set and return a metrics report dict.

    Also prints a summary table to stdout and writes per-record rows to
    ``results_path`` (eval/results.csv by default).
    """
    if not _has_api_key():
        print(
            "!! No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY).\n"
            "   The pipeline needs the LLM touchpoints. Set a key and retry."
        )
        raise SystemExit(1)

    gt = _load_rows(ids, limit)
    if gt.empty:
        print("No matching ground-truth rows to evaluate.")
        raise SystemExit(1)

    logger = AuditLogger(path=EVAL_AUDIT_LOG_PATH, reset=True)
    print(f"Ground truth : {GROUND_TRUTH_PATH}")
    print(f"Audit log    : {logger.path}")
    print("Building RetrievalService (loading model + encoding catalogue)...\n")

    from services.retrieval_service import RetrievalService

    retrieval = RetrievalService()
    app = build_graph(audit_logger=logger)

    results: list[dict] = []
    for _, row in gt.iterrows():
        row_d = row.to_dict()
        input_record = {
            "input_id": row_d["input_id"],
            "raw_input": row_d["raw_input"],
        }
        print(
            f"  running {row_d['input_id']:<6} [{row_d['expected_match_type']:<9}] "
            f"{row_d['raw_input']!r}"
        )
        state = _run_record(app, retrieval, input_record)
        results.append(_score_record(row_d, state))

    print()
    print("=" * 78)
    print(f"EVALUATION SUMMARY  ({len(results)} records)")
    print("=" * 78)
    _print_accuracy(results)
    _print_auto_approve_precision(results)
    _print_cost_latency(results)
    _print_step_distribution(results)

    _write_results_csv(results, results_path)
    print(f"Per-record results written to: {results_path}")
    print("=" * 78)

    # --- assemble the machine-readable report -----------------------------
    n = len(results)
    without_t4 = [r for r in results if r["reached_verifier"]]
    with_t4 = [r for r in results if r["predicted_decision"] == "approved"]

    def _prec(approved: list[dict]) -> Optional[float]:
        if not approved:
            return None
        correct = sum(
            r["expected_match_type"] == "match" and r["id_correct"] for r in approved
        )
        return correct / len(approved)

    downgraded = [r for r in without_t4 if r["downgraded_by_verifier"]]

    return {
        "n_records": n,
        "strict_accuracy": sum(r["correct"] for r in results) / n,
        "tier_accuracy": sum(r["decision_correct"] for r in results) / n,
        "auto_approve_precision_without_t4": _prec(without_t4),
        "auto_approve_precision_with_t4": _prec(with_t4),
        "n_auto_approved_without_t4": len(without_t4),
        "n_auto_approved_with_t4": len(with_t4),
        "verifier_downgrade_rate": (len(downgraded) / len(without_t4)) if without_t4 else None,
        "step_count_distribution": dict(Counter(r["step_count"] for r in results)),
        "results_path": str(results_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the MMV matching pipeline.")
    parser.add_argument("--ids", default="", help="comma-separated input_ids (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    args = parser.parse_args()

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
    evaluate(ids=ids, limit=args.limit)


if __name__ == "__main__":
    main()
