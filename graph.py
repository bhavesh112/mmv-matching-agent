"""LangGraph orchestration for the MMV matching pipeline.

Assembles the nodes (normalization -> reformulation -> retrieval -> ReAct
reasoning with tools -> confidence routing -> adversarial verification ->
explanation -> audit) into a StateGraph over MMVState.

The core is a confidence router between the ReAct reasoning node (Touchpoint 3)
and the adversarial verifier (Touchpoint 4): it buckets the reasoning agent's
confidence into three tiers and only spends an LLM verifier call on the top one.
Whichever tier a record lands in, it converges on a terminal decision node that
stamps ``final_decision``, then flows through the explanation touchpoint and the
audit logger before ending:

    reason ─┬─ confidence  > 0.95 ────────────► verify ─┬─ pass ─► approve ┐
            │                                            └─ fail ─► review ┤
            ├─ 0.80 ≤ confidence ≤ 0.95 ─────────────────────────► review ┼─► explain ─► audit ─► END
            └─ confidence < 0.80 ────────────────────────────────► reject ┘

Thresholds live in ``config`` (AUTO_APPROVE_CONFIDENCE_THRESHOLD /
REVIEW_CONFIDENCE_FLOOR). The verifier can only DOWNGRADE an auto-approval to
review — a lone adversarial objection against a very high-confidence match
warrants a human look, not an automatic rejection.

``explain`` (Touchpoint 5) synthesizes the whole trace into a reviewer-readable
summary; ``audit`` persists the full state + final_decision + explanation as one
JSON-line record via the ``AuditLogger`` — so every record that traverses the
graph leaves a complete audit entry, no matter which tier it took.

The reasoning node consumes ``candidate_records`` produced by Phase 2 retrieval,
so callers populate that field (plus ``normalized_record``) on the input state.
The remaining touchpoints (normalization, reformulation) run ahead of this graph
in the driver (see app.py) and slot in as real nodes as they are wired.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

import config
from llm_touchpoints.explanation_llm import explain
from llm_touchpoints.reasoning_llm import reason
from llm_touchpoints.verifier_llm import verify
from logging_config import get_logger
from services.audit_logger import AuditLogger
from state import MMVState

log = get_logger("graph")


def _route_by_confidence(state: MMVState) -> str:
    """Bucket the reasoning confidence into the three routing tiers."""
    confidence = float(state.get("confidence") or 0.0)
    if confidence > config.AUTO_APPROVE_CONFIDENCE_THRESHOLD:
        route = "verify"
    elif confidence >= config.REVIEW_CONFIDENCE_FLOOR:
        route = "review"
    else:
        route = "reject"
    log.info("route: reason conf=%.2f -> %s", confidence, route)
    return route


def _route_after_verify(state: MMVState) -> str:
    """A passing verifier auto-approves; any failure downgrades to review."""
    result = state.get("verifier_result") or {}
    route = "approve" if result.get("passed") else "review"
    log.info("route: verify passed=%s -> %s", bool(result.get("passed")), route)
    return route


def _approve(state: MMVState) -> dict:
    return {"final_decision": "approved"}


def _review(state: MMVState) -> dict:
    return {"final_decision": "review"}


def _reject(state: MMVState) -> dict:
    return {"final_decision": "rejected"}


def _make_audit_node(logger: AuditLogger):
    """Terminal side-effect node: persist the completed record, update nothing.

    Runs after ``explain`` so the audit entry captures the final_decision AND the
    explanation. Returns an empty update — the graph's job here is the write."""

    def audit(state: MMVState) -> dict:
        logger.log(state)
        return {}

    return audit


def build_graph(audit_logger: Optional[AuditLogger] = None):
    """Construct and return the compiled LangGraph app.

    Wiring: ``START -> reason -> {verify | review | reject}``; ``verify ->
    {approve | review}``. The approve/review/reject terminal decision nodes stamp
    ``final_decision`` and converge on ``explain -> audit -> END``, so every
    record produces a final_decision, an explanation, and an audit log entry.

    ``audit_logger`` lets the caller (e.g. app.py / eval) own the log-file
    lifecycle — path, append vs. reset, reading entries back; a default one
    (writing to ``config.AUDIT_LOG_PATH``) is created if none is passed.
    """
    logger = audit_logger or AuditLogger()

    graph = StateGraph(MMVState)

    graph.add_node("reason", reason)
    graph.add_node("verify", verify)
    graph.add_node("approve", _approve)
    graph.add_node("review", _review)
    graph.add_node("reject", _reject)
    graph.add_node("explain", explain)
    graph.add_node("audit", _make_audit_node(logger))

    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        _route_by_confidence,
        {"verify": "verify", "review": "review", "reject": "reject"},
    )
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"approve": "approve", "review": "review"},
    )
    # All three decision tiers converge on the explanation, then the audit sink.
    graph.add_edge("approve", "explain")
    graph.add_edge("review", "explain")
    graph.add_edge("reject", "explain")
    graph.add_edge("explain", "audit")
    graph.add_edge("audit", END)

    return graph.compile()
