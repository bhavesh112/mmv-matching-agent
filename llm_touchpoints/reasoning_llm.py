"""Touchpoint 3 — ReAct reasoning agent (LangGraph).

The core disambiguation step. Given the normalized input and a ranked list of
candidate records (from Phase 2 retrieval), a ReAct agent reasons step by step,
calling deterministic tools to converge on a single ``selected_match`` (or a
reject):

  * ``attribute_compare`` — per-attribute diff of a candidate vs the input.
  * ``validate``          — hard-rule gate (fuel / transmission / cc / seating /
                            year); pure Python, no LLM.
  * ``broaden_candidates``— re-run retrieval with a larger / different query when
                            none of the current candidates look right.
  * ``finish``            — commit to {match, confidence, reason}.

The loop is a small LangGraph ``StateGraph`` (agent node -> tool node -> loop),
which gives us three guarantees the raw LLM cannot be trusted to keep:

  1. **Step cap.** After ``config.MAX_REASONING_STEPS`` agent turns the loop is
     force-terminated and the decision escalated to review.
  2. **Validation gate.** An "approved"-eligible confidence
     (>= ``APPROVE_CONFIDENCE_THRESHOLD``) is only allowed to survive if
     ``validate`` PASSED for the selected candidate; otherwise it is capped into
     the review band.
  3. **Full audit trail.** Every thought / action / observation is recorded in
     ``reasoning_trace`` for logging.

Public surface:
- ``reason(state)`` — LangGraph node: reads MMVState, returns a partial update.
- ``run_reasoning_agent(normalized_record, candidates, ...)`` — helper that
  returns the full decision + trace for scripts/tests without an MMVState.
"""

from __future__ import annotations

import json
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from logging_config import get_logger
from prompts.react_reasoning_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from services.llm_service import LLMService
from state import MMVState
from tools.attribute_compare_tool import attribute_compare
from tools.finish_tool import finish
from tools.validate_tool import validate

# Candidate fields surfaced to the model (kept compact; the tools see the full
# record). year is shown as a [start, end] range.
_CANDIDATE_FIELDS = (
    "mmv_id",
    "make",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "cc",
    "seating_capacity",
    "year_start",
    "year_end",
    "combined_score",
)

# Signature of the retrieval callback used by broaden_candidates.
RetrieveFn = Callable[[str, int], list]


# --- ReAct subgraph state -------------------------------------------------


class _ReActState(TypedDict):
    normalized_record: dict
    candidates: list
    scratchpad: list          # list of trace entries -> becomes reasoning_trace
    step_count: int
    pending_action: dict       # {action, action_input} chosen by the agent node
    validations: dict          # mmv_id -> validate() result
    finished: bool
    result: dict               # {match, confidence, reason} (final)
    validation_status: str


# --- helpers --------------------------------------------------------------


def _index_by_id(candidates: list) -> dict:
    return {c.get("mmv_id"): c for c in candidates if c.get("mmv_id")}


def _slim_candidate(candidate: dict) -> dict:
    return {k: candidate.get(k) for k in _CANDIDATE_FIELDS if k in candidate}


def _reconstruct_query(normalized_record: dict) -> str:
    """Best-effort free-text query from a normalized record (for broaden)."""
    if normalized_record.get("raw_input"):
        return str(normalized_record["raw_input"])
    parts = [
        str(normalized_record[k])
        for k in ("make", "model", "variant", "fuel_type", "transmission")
        if normalized_record.get(k)
    ]
    return " ".join(parts)


def _best_candidate_id(candidates: list) -> Optional[str]:
    """The top-ranked candidate id, used when escalation needs a best guess."""
    if not candidates:
        return None
    return candidates[0].get("mmv_id")


def _finalize(
    state: _ReActState,
    match_id: Optional[str],
    confidence: float,
    reason: str,
    *,
    forced_review: bool = False,
) -> dict:
    """Apply the validation gate + review cap and build the final result dict."""
    index = _index_by_id(state["candidates"])
    match_record = index.get(match_id) if match_id else None
    validation = state["validations"].get(match_id) if match_id else None
    validated_pass = bool(validation and validation.get("validation_status") == "pass")

    notes: list[str] = []
    conf = max(0.0, min(1.0, float(confidence)))

    if forced_review:
        conf = min(conf, config.REVIEW_CONFIDENCE_CAP)
        notes.append(
            "escalated to review: step budget exhausted before a confident decision"
        )

    # Gate: a real match can only stay approved-eligible if validate passed.
    if (
        match_id
        and conf >= config.APPROVE_CONFIDENCE_THRESHOLD
        and not validated_pass
    ):
        conf = min(conf, config.REVIEW_CONFIDENCE_CAP)
        notes.append(
            "confidence capped below approval threshold: validate_tool did not "
            "pass for the selected candidate"
        )

    final_reason = reason.strip()
    if notes:
        final_reason = (final_reason + " " if final_reason else "") + " ".join(
            f"[{n}]" for n in notes
        )

    result = finish(match_record, conf, final_reason)

    if match_id is None:
        validation_status = "reject"
    elif validation is None:
        validation_status = "not_validated"
    else:
        validation_status = validation.get("validation_status", "unknown")

    return {
        "finished": True,
        "result": result,
        "validation_status": validation_status,
    }


# --- graph nodes ----------------------------------------------------------


def _make_agent_node(service: LLMService):
    def agent_node(state: _ReActState) -> dict:
        step = state["step_count"] + 1

        # Step-cap: force escalation to review before spending another LLM call.
        if step > config.MAX_REASONING_STEPS:
            best = _best_candidate_id(state["candidates"])
            entry = {
                "step": step,
                "thought": (
                    "Step budget exhausted; escalating to human review with the "
                    "best available candidate."
                ),
                "action": "finish",
                "action_input": {"match": best, "confidence": 0.0},
                "observation": None,
            }
            finalize = _finalize(
                state,
                best,
                0.0,
                "Escalated to review after exhausting the reasoning step budget.",
                forced_review=True,
            )
            entry["observation"] = finalize["result"]["reason"]
            return {
                "step_count": step,
                "scratchpad": state["scratchpad"] + [entry],
                "pending_action": {"action": "finish", "action_input": {}},
                **finalize,
            }

        candidates_display = json.dumps(
            [_slim_candidate(c) for c in state["candidates"]],
            ensure_ascii=False,
            indent=2,
        )
        scratchpad_display = (
            json.dumps(state["scratchpad"], ensure_ascii=False, indent=2)
            if state["scratchpad"]
            else "(empty — this is the first step)"
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(
            normalized_record=json.dumps(
                state["normalized_record"], ensure_ascii=False, indent=2
            ),
            candidates=candidates_display,
            scratchpad=scratchpad_display,
            step=step,
            max_steps=config.MAX_REASONING_STEPS,
        )

        try:
            data = service.generate_json("reasoning", SYSTEM_PROMPT, user_prompt)
        except (ValueError, RuntimeError) as exc:
            # Unparseable / failed LLM turn: escalate rather than crash the graph.
            best = _best_candidate_id(state["candidates"])
            entry = {
                "step": step,
                "thought": f"Agent step failed: {exc}",
                "action": "finish",
                "action_input": {"match": best, "confidence": 0.0},
                "observation": "escalated to review (agent error)",
            }
            finalize = _finalize(
                state,
                best,
                0.0,
                f"Escalated to review after an agent error: {exc}",
                forced_review=True,
            )
            return {
                "step_count": step,
                "scratchpad": state["scratchpad"] + [entry],
                "pending_action": {"action": "finish", "action_input": {}},
                **finalize,
            }

        action = str(data.get("action", "")).strip()
        action_input = data.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {}
        entry = {
            "step": step,
            "thought": data.get("thought", ""),
            "action": action,
            "action_input": action_input,
            "observation": None,
        }
        return {
            "step_count": step,
            "scratchpad": state["scratchpad"] + [entry],
            "pending_action": {"action": action, "action_input": action_input},
        }

    return agent_node


def _make_tool_node(retrieve_fn: Optional[RetrieveFn]):
    def tool_node(state: _ReActState) -> dict:
        action = state["pending_action"].get("action", "")
        action_input = state["pending_action"].get("action_input", {})
        index = _index_by_id(state["candidates"])
        # The trace entry for the current step is the last one; attach observation.
        scratchpad = [dict(e) for e in state["scratchpad"]]
        current = scratchpad[-1] if scratchpad else None

        update: dict = {}

        if action == "finish":
            match_id = action_input.get("match")
            # Treat empty string / "null" / "none" as a reject.
            if isinstance(match_id, str) and match_id.strip().lower() in (
                "",
                "null",
                "none",
            ):
                match_id = None
            confidence = action_input.get("confidence", 0.0)
            reason = action_input.get("reason", "")
            finalize = _finalize(state, match_id, confidence, reason)
            if current is not None:
                current["observation"] = finalize["result"]["reason"]
            update.update(finalize)
            update["scratchpad"] = scratchpad
            return update

        if action == "attribute_compare":
            mmv_id = action_input.get("mmv_id")
            candidate = index.get(mmv_id)
            if candidate is None:
                observation = {
                    "error": f"unknown mmv_id '{mmv_id}'; "
                    f"choose from {sorted(index)}"
                }
            else:
                observation = attribute_compare(state["normalized_record"], candidate)

        elif action == "validate":
            mmv_id = action_input.get("mmv_id")
            candidate = index.get(mmv_id)
            if candidate is None:
                observation = {
                    "error": f"unknown mmv_id '{mmv_id}'; "
                    f"choose from {sorted(index)}"
                }
            else:
                observation = validate(state["normalized_record"], candidate)
                validations = dict(state["validations"])
                validations[mmv_id] = observation
                update["validations"] = validations

        elif action == "broaden_candidates":
            if retrieve_fn is None:
                observation = {"error": "broaden_candidates is unavailable here"}
            else:
                query = action_input.get("query") or _reconstruct_query(
                    state["normalized_record"]
                )
                top_n = action_input.get("top_n") or config.BROADEN_TOP_N
                try:
                    top_n = int(top_n)
                except (TypeError, ValueError):
                    top_n = config.BROADEN_TOP_N
                new_candidates = retrieve_fn(query, top_n)
                merged = _index_by_id(state["candidates"])
                added = 0
                for cand in new_candidates:
                    cid = cand.get("mmv_id")
                    if cid and cid not in merged:
                        merged[cid] = cand
                        added += 1
                candidates = list(merged.values())
                update["candidates"] = candidates
                observation = {
                    "query": query,
                    "top_n": top_n,
                    "added": added,
                    "total_candidates": len(candidates),
                    "candidates": [_slim_candidate(c) for c in candidates],
                }

        else:
            observation = {
                "error": f"unknown action '{action}'. Valid actions: "
                "attribute_compare, validate, broaden_candidates, finish."
            }

        if current is not None:
            current["observation"] = observation
        update["scratchpad"] = scratchpad
        return update

    return tool_node


def _route_after_agent(state: _ReActState) -> str:
    return END if state.get("finished") else "tools"


def _route_after_tools(state: _ReActState) -> str:
    return END if state.get("finished") else "agent"


def build_reasoning_agent(
    service: LLMService, retrieve_fn: Optional[RetrieveFn] = None
):
    """Compile and return the ReAct StateGraph for the reasoning touchpoint."""
    graph = StateGraph(_ReActState)
    graph.add_node("agent", _make_agent_node(service))
    graph.add_node("tools", _make_tool_node(retrieve_fn))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", _route_after_agent, {"tools": "tools", END: END}
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools, {"agent": "agent", END: END}
    )
    return graph.compile()


# --- public entrypoints ---------------------------------------------------


def run_reasoning_agent(
    normalized_record: dict,
    candidates: list,
    service: Optional[LLMService] = None,
    retrieve_fn: Optional[RetrieveFn] = None,
) -> dict:
    """Run the ReAct agent end to end and return the full decision + trace.

    Returns ``{"match", "confidence", "reason", "reasoning_trace",
    "validation_status", "step_count", "llm_call_log"}``. ``match`` is the full
    candidate record (or ``None`` for a reject).
    """
    service = service or LLMService()
    agent = build_reasoning_agent(service, retrieve_fn)

    initial: _ReActState = {
        "normalized_record": normalized_record,
        "candidates": list(candidates),
        "scratchpad": [],
        "step_count": 0,
        "pending_action": {},
        "validations": {},
        "finished": False,
        "result": {},
        "validation_status": "not_validated",
    }
    final = agent.invoke(
        initial, {"recursion_limit": config.REASONING_RECURSION_LIMIT}
    )

    result = final.get("result") or {"match": None, "confidence": 0.0, "reason": ""}
    return {
        "match": result.get("match"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "reasoning_trace": final.get("scratchpad", []),
        "validation_status": final.get("validation_status", "not_validated"),
        "step_count": final.get("step_count", 0),
        "llm_call_log": list(service.call_log),
    }


# Lazily-built retrieval service, so the reasoning node only pays the (heavy)
# embedding/catalogue load if the agent actually asks to broaden candidates.
_retrieval_service = None


def _default_retrieve_fn(query: str, top_n: int) -> list:
    global _retrieval_service
    if _retrieval_service is None:
        from services.retrieval_service import RetrievalService

        _retrieval_service = RetrievalService()
    return _retrieval_service.retrieve(query, top_n=top_n)


def reason(state: MMVState) -> dict:
    """LangGraph node: run the ReAct agent over Phase 2's candidate_records.

    Returns a partial MMVState update: ``selected_match``, ``confidence``,
    ``reasoning_decision`` ({match, confidence, reason}), ``reasoning_trace``,
    ``validation_status``, ``step_count`` and ``llm_call_log``.
    """
    log = get_logger("node.reason")
    service = LLMService()
    normalized = state.get("normalized_record") or state.get("input_record") or {}
    candidates = state.get("candidate_records") or []
    log.info("reason start: %d candidates", len(candidates))

    outcome = run_reasoning_agent(
        normalized,
        candidates,
        service=service,
        retrieve_fn=_default_retrieve_fn,
    )

    match = outcome["match"] or {}
    log.info(
        "reason done: match=%s conf=%.2f steps=%d validation=%s",
        match.get("mmv_id", "none"), outcome["confidence"],
        outcome["step_count"], outcome["validation_status"],
    )

    return {
        "selected_match": outcome["match"] or {},
        "confidence": outcome["confidence"],
        "reasoning_decision": {
            "match": outcome["match"],
            "confidence": outcome["confidence"],
            "reason": outcome["reason"],
        },
        "reasoning_trace": outcome["reasoning_trace"],
        "validation_status": outcome["validation_status"],
        "step_count": outcome["step_count"],
        "llm_call_log": outcome["llm_call_log"],
    }
