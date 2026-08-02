"""Shared graph state for the MMV matching pipeline.

MMVState is the single object threaded through the LangGraph. Each of the six
LLM touchpoints and the deterministic tools read from / write to these fields.
"""

from __future__ import annotations

from typing import TypedDict


class MMVState(TypedDict):
    input_record: dict
    normalized_record: dict       # output of Touchpoint 1
    search_queries: list          # output of Touchpoint 2
    candidate_records: list
    selected_match: dict
    reasoning_trace: list         # ReAct scratchpad for Touchpoint 3
    reasoning_decision: dict      # {match, confidence, reason} from Touchpoint 3
    validation_status: str
    verifier_result: dict         # output of Touchpoint 4 (pass/fail + concerns raised)
    confidence: float
    explanation: str              # output of Touchpoint 5
    final_decision: str           # "approved" | "review" | "rejected"
    step_count: int
    feedback: str
    llm_call_log: list            # which touchpoints fired, cost, latency — for eval
