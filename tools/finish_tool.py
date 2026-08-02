"""Finish tool — terminates the ReAct loop.

The agent calls this once it has committed to a selected_match (or a reject),
signalling the graph to move on to verification/explanation. It packages the
final ``{match, confidence, reason}`` decision into a stable shape; a ``None``
match means the agent chose to reject (no catalogue entry fits).
"""

from __future__ import annotations

from typing import Optional


def finish(
    selected_match: Optional[dict], confidence: float, reason: str = ""
) -> dict:
    """Signal completion with the chosen match, a confidence score and a reason.

    ``selected_match`` is the full candidate record (carrying ``mmv_id``) or
    ``None`` for a reject. ``confidence`` is clamped to [0, 1].
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        "match": selected_match or None,
        "confidence": conf,
        "reason": reason or "",
    }
