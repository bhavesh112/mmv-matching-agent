"""Touchpoint 4 — Adversarial verifier.

Independently challenges a high-confidence ``selected_match`` before it is
allowed to auto-approve. Runs ONLY on the top routing tier (reasoning confidence
> ``config.AUTO_APPROVE_CONFIDENCE_THRESHOLD``); the review / reject tiers never
reach it.

The check is deliberately adversarial and runs at a hotter temperature (and,
optionally, on a different model) than the reasoning agent — see
``prompts/verifier_prompt.py`` and the ``VERIFIER_*`` knobs in ``config`` — so it
does not share the reasoning agent's blind spots and re-derive the same answer.

Output (``verifier_result``): ``{passed, verdict, concern, ok}``.
  * ``verdict`` — "pass" (match survived scrutiny) or "fail" (a substantive
    concern was raised). ``passed`` is the boolean form.
  * ``concern`` — the verifier's stated strongest argument against the match.
  * ``ok`` — whether the verifier LLM call itself succeeded; on error the result
    fails closed (``passed=False``) so a broken verifier never auto-approves.

A "fail" routes the record to MANUAL REVIEW (never to reject): the reasoning
agent was highly confident, so a lone adversarial objection warrants a human
look, not an automatic rejection.
"""

from __future__ import annotations

import json
from typing import Optional

import config
from logging_config import get_logger
from prompts.verifier_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from services.llm_service import LLMService
from state import MMVState

# Retrieval bookkeeping fields — noise to the verifier, which reasons about
# vehicle attributes, so we strip them from the record shown in the prompt.
_SCORE_FIELDS = ("fuzzy_score", "embedding_score", "combined_score")


def _slim_match(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _SCORE_FIELDS}


def _make_verifier_service() -> LLMService:
    """LLMService configured for the adversarial verifier (hotter / distinct)."""
    return LLMService(
        model=config.VERIFIER_MODEL,
        temperature=config.VERIFIER_TEMPERATURE,
    )


def run_verifier(
    normalized_record: dict,
    selected_match: dict,
    reason: str = "",
    confidence: Optional[float] = None,
    service: Optional[LLMService] = None,
) -> dict:
    """Adversarially challenge ``selected_match``; return the verifier result.

    Returns ``{"passed", "verdict", "concern", "ok", "llm_call_log"}``. Fails
    closed (``passed=False``) if the match is empty or the LLM call/parse fails,
    so a verifier problem can only downgrade a match to review — never wave one
    through.
    """
    if not selected_match:
        return {
            "passed": False,
            "verdict": "fail",
            "concern": "no selected_match to verify",
            "ok": False,
            "llm_call_log": [],
        }

    own_service = service is None
    service = service or _make_verifier_service()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        normalized_record=json.dumps(normalized_record, ensure_ascii=False, indent=2),
        selected_match=json.dumps(_slim_match(selected_match), ensure_ascii=False, indent=2),
        confidence="unknown" if confidence is None else confidence,
        reason=(reason or "").strip() or "(no justification provided)",
    )

    log = list(service.call_log) if not own_service else None
    try:
        data = service.generate_json("verifier", SYSTEM_PROMPT, user_prompt)
    except (ValueError, RuntimeError) as exc:
        # A failed/unparseable verifier turn must NOT auto-approve. Fail closed.
        return {
            "passed": False,
            "verdict": "fail",
            "concern": f"verifier error: {exc}",
            "ok": False,
            "llm_call_log": _new_calls(service, log),
        }

    verdict = str(data.get("verdict", "")).strip().lower()
    passed = verdict == "pass"
    concern = str(data.get("concern", "")).strip()
    return {
        "passed": passed,
        "verdict": "pass" if passed else "fail",
        "concern": concern,
        "ok": True,
        "llm_call_log": _new_calls(service, log),
    }


def _new_calls(service: LLMService, prior_len_marker: Optional[list]) -> list:
    """Calls this run added to ``service`` (all of them if we own the service)."""
    if prior_len_marker is None:
        return list(service.call_log)
    return service.call_log[len(prior_len_marker):]


def verify(state: MMVState) -> dict:
    """LangGraph node: run the adversarial verifier over the selected match.

    Returns a partial MMVState update: ``verifier_result`` plus the verifier's
    LLM call(s) appended to ``llm_call_log``. Only invoked on the top routing
    tier (see ``graph.build_graph``).
    """
    log = get_logger("node.verify")
    normalized = state.get("normalized_record") or state.get("input_record") or {}
    selected = state.get("selected_match") or {}
    decision = state.get("reasoning_decision") or {}
    log.info("verify start: match=%s", selected.get("mmv_id", "none"))

    outcome = run_verifier(
        normalized,
        selected,
        reason=decision.get("reason", ""),
        confidence=state.get("confidence"),
    )
    log.info(
        "verify done: passed=%s verdict=%s",
        outcome["passed"], outcome.get("verdict", "?"),
    )

    verifier_result = {
        "passed": outcome["passed"],
        "verdict": outcome["verdict"],
        "concern": outcome["concern"],
        "ok": outcome["ok"],
    }
    prior_log = state.get("llm_call_log") or []
    return {
        "verifier_result": verifier_result,
        "llm_call_log": list(prior_log) + outcome.get("llm_call_log", []),
    }
