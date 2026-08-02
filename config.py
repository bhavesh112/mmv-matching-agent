"""Central configuration for the MMV matching pipeline.

Tunables that govern control flow (rather than data) live here so they can be
adjusted in one place and referenced by name from the touchpoints. Env vars
override the defaults where noted, keeping deployments configurable without code
changes.
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- Touchpoint 3: ReAct reasoning agent ----------------------------------

# Maximum number of reasoning steps (agent turns) before the loop is force-
# terminated and the decision is escalated to human review. Prevents the agent
# from looping indefinitely and bounds cost per record.
MAX_REASONING_STEPS: int = _int_env("MMV_MAX_REASONING_STEPS", 6)

# Confidence at or above which a match is eligible to be auto-"approved"
# downstream. The agent may only emit a confidence >= this threshold when
# validate_tool has PASSED for the selected candidate; otherwise the confidence
# is capped just below it, forcing the record into the "review" band.
APPROVE_CONFIDENCE_THRESHOLD: float = _float_env(
    "MMV_APPROVE_CONFIDENCE_THRESHOLD", 0.85
)

# Confidence assigned when a decision is escalated to review (step cap hit, or an
# approved-eligible confidence was requested without a passing validation). Sits
# below APPROVE_CONFIDENCE_THRESHOLD by construction.
REVIEW_CONFIDENCE_CAP: float = _float_env("MMV_REVIEW_CONFIDENCE_CAP", 0.6)

# Candidate-set size used when the agent asks to broaden the search because none
# of the current candidates look right.
BROADEN_TOP_N: int = _int_env("MMV_BROADEN_TOP_N", 15)


# --- Graph-level confidence routing (post-reasoning) ----------------------

# The reasoning agent's confidence (0..1) is bucketed into three routing tiers
# after Touchpoint 3, matching the ">95 / 80-95 / <80" policy:
#
#   confidence  > AUTO_APPROVE_CONFIDENCE_THRESHOLD  -> adversarial VERIFIER,
#       then auto-"approved" only if the verifier passes (fail -> review).
#   REVIEW_CONFIDENCE_FLOOR <= confidence <= AUTO_APPROVE_CONFIDENCE_THRESHOLD
#       -> manual "review" (never auto-approved, never auto-rejected).
#   confidence  < REVIEW_CONFIDENCE_FLOOR            -> "rejected".
#
# Note these are strictly above the reasoning touchpoint's own approval gate
# (APPROVE_CONFIDENCE_THRESHOLD): a confidence can only exceed 0.95 if validate
# passed, so only validated matches ever reach the verifier.
AUTO_APPROVE_CONFIDENCE_THRESHOLD: float = _float_env(
    "MMV_AUTO_APPROVE_CONFIDENCE_THRESHOLD", 0.95
)
REVIEW_CONFIDENCE_FLOOR: float = _float_env("MMV_REVIEW_CONFIDENCE_FLOOR", 0.80)


# --- Touchpoint 4: adversarial verifier -----------------------------------

# The verifier deliberately runs hotter than the reasoning agent (which runs at
# temperature 0.0) and, optionally, on a different model, so the two do not share
# blind spots — the whole point of an independent adversarial check.
VERIFIER_TEMPERATURE: float = _float_env("MMV_VERIFIER_TEMPERATURE", 0.7)

# Optional distinct model for the verifier. None -> use the LLMService default
# (GEMINI_MODEL / gemini-flash-latest), just at the hotter temperature above.
VERIFIER_MODEL = os.environ.get("MMV_VERIFIER_MODEL") or None

# LangGraph recursion limit for the ReAct subgraph. Each reasoning step spans an
# agent node + a tool node, so the graph needs headroom above MAX_REASONING_STEPS.
REASONING_RECURSION_LIMIT: int = _int_env(
    "MMV_REASONING_RECURSION_LIMIT", 2 * MAX_REASONING_STEPS + 5
)


# --- Touchpoint 5: explanation --------------------------------------------

# The explanation is a summarization-only step (no reasoning, no tools), so it
# runs on the CHEAPEST/smallest tier — flash-lite — rather than the flash model
# the reasoning agent and verifier use. Override with MMV_EXPLANATION_MODEL.
EXPLANATION_MODEL = os.environ.get("MMV_EXPLANATION_MODEL") or "gemini-flash-lite-latest"

# Low but non-zero temperature: we want a fluent reviewer-readable sentence or
# two, not creative embellishment of the decision it is summarizing.
EXPLANATION_TEMPERATURE: float = _float_env("MMV_EXPLANATION_TEMPERATURE", 0.2)


# --- Audit log ------------------------------------------------------------

# Where the per-record audit trail (full state + final_decision + explanation)
# is written, as JSON lines. Relative paths resolve against the project root.
# Override with MMV_AUDIT_LOG_PATH.
AUDIT_LOG_PATH = os.environ.get("MMV_AUDIT_LOG_PATH") or "logs/audit_log.jsonl"


# --- Operational logging --------------------------------------------------

# Where the backend's operational log (job lifecycle, node steps, LLM calls) is
# written, with rotation. Relative paths resolve against the project root. This
# is distinct from AUDIT_LOG_PATH (the per-record domain audit trail).
# Override with MMV_LOG_PATH / MMV_LOG_LEVEL.
LOG_PATH = os.environ.get("MMV_LOG_PATH") or "logs/backend.log"
LOG_LEVEL = os.environ.get("MMV_LOG_LEVEL") or "INFO"
