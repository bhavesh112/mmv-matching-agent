"""Audit logger.

Persists one structured record per processed input — the FULL decision trace plus
the final outcome — so every pipeline run leaves a complete, replayable paper
trail. Consumers: the eval harness, human reviewers auditing "review"/"rejected"
records, and the Touchpoint 6 feedback miner.

Format: JSON Lines (one self-contained JSON object per line). This is chosen over
SQLite deliberately — the write path is pure append, the records are heterogeneous
nested documents (traces, call logs), and downstream tooling (pandas, jq, the
feedback miner) reads them as a stream. No schema migrations, trivially greppable.

Each entry captures: the input, the normalized record, the candidates considered,
the reasoning decision + full ReAct trace, the validation status, the verifier
outcome, the final_decision, the human-readable explanation, and the per-touchpoint
llm_call_log (cost/latency) — timestamped and keyed by a record id.

Public surface:
- ``AuditLogger`` — open once per run; call ``.log(state)`` per record. Append
  mode by default; pass ``reset=True`` to start a fresh log file.
- ``AuditLogger.read_all()`` — read the written entries back (for confirmation /
  eval).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from state import MMVState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fields copied verbatim from MMVState into every audit entry. Kept explicit (vs.
# dumping the whole state) so the log shape is stable and intentional.
_TRACE_FIELDS = (
    "input_record",
    "normalized_record",
    "search_queries",
    "selected_match",
    "reasoning_decision",
    "reasoning_trace",
    "validation_status",
    "verifier_result",
    "confidence",
    "explanation",
    "final_decision",
    "step_count",
    "llm_call_log",
)


def _resolve_path(path: Optional[str]) -> Path:
    """Resolve a configured/overridden path against the project root."""
    raw = Path(path or config.AUDIT_LOG_PATH)
    return raw if raw.is_absolute() else (_PROJECT_ROOT / raw)


def _candidate_ids(state: MMVState) -> list:
    """Just the ids of the candidates considered — the full records live in the
    reasoning trace / selected_match, so the entry stays compact."""
    return [
        c.get("mmv_id")
        for c in (state.get("candidate_records") or [])
        if isinstance(c, dict) and c.get("mmv_id")
    ]


def _record_id(state: MMVState) -> str:
    """Best-effort stable id for the record: input_id if the caller set one,
    else the raw input string, else a generated uuid."""
    src = state.get("input_record") or {}
    for key in ("input_id", "id"):
        if src.get(key):
            return str(src[key])
    raw = src.get("raw_input") or (state.get("normalized_record") or {}).get("raw_input")
    if raw:
        return str(raw)
    return f"anon-{uuid.uuid4().hex[:8]}"


class AuditLogger:
    """Appends one JSON-line audit entry per processed record."""

    def __init__(self, path: Optional[str] = None, reset: bool = False) -> None:
        self.path = _resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        # Touch the file so a run always produces a log, even with zero records.
        self.path.touch(exist_ok=True)

    def build_entry(self, state: MMVState) -> dict:
        """Assemble (but do not write) the structured audit entry for a record."""
        entry: dict = {
            "audit_id": uuid.uuid4().hex,
            "record_id": _record_id(state),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "candidate_ids": _candidate_ids(state),
        }
        for key in _TRACE_FIELDS:
            entry[key] = state.get(key)
        return entry

    def log(self, state: MMVState) -> dict:
        """Build the entry, append it as one JSON line, and return it."""
        entry = self.build_entry(state)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return entry

    def read_all(self) -> list[dict]:
        """Read every entry written so far back into memory (skips blank lines)."""
        if not self.path.exists():
            return []
        entries: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
