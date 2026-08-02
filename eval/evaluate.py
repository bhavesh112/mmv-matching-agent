"""Evaluation harness for the MMV matching pipeline.

Runs the graph over eval/labeled_ground_truth.csv and reports:
  - overall accuracy
  - precision by tier (easy / ambiguous / reject)
  - cost and latency per touchpoint (from llm_call_log)

Stub for now.
"""

from __future__ import annotations

from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "labeled_ground_truth.csv"


def evaluate() -> dict:
    """Run the pipeline over the labeled set and return a metrics report."""
    raise NotImplementedError("evaluate not implemented yet")


if __name__ == "__main__":
    evaluate()
