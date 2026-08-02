"""Thin FastAPI layer over the graph.py matching pipeline.

Exposes the pipeline as an HTTP API a browser UI can drive:

    POST /api/match           upload a master + input CSV, kick off a run
    GET  /api/jobs/{id}       poll a run's status + per-record results/traces
    POST /api/jobs/{id}/records/{rid}/override   a human overrides a decision
    GET  /api/health          key presence, model, routing thresholds
    GET  /api/sample          the bundled ground-truth inputs (for a demo run)

A run is a background job (thread) writing per-record results into an in-memory
store as they finish, so the UI can stream progress by polling. Each record is
flattened into the touchpoint-trace shape by ``services.matching_pipeline`` —
identical whether it came from the live (Gemini) pipeline or the deterministic
mock pipeline.

Run it with:  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env before importing anything that reads the API key.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import config  # noqa: E402
from logging_config import get_logger, setup_logging  # noqa: E402
from services.csv_loader import load_mmv_master  # noqa: E402
from services.matching_pipeline import (  # noqa: E402
    MockPipeline,
    build_record_result,
    thresholds,
)

setup_logging()
log = get_logger("server")

_PROJECT_ROOT = Path(__file__).resolve().parent
_SAMPLE_INPUTS = _PROJECT_ROOT / "eval" / "labeled_ground_truth.csv"
CANDIDATE_TOP_N = 8

app = FastAPI(title="MMV Matching API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory job store (single-process; fine for a review/demo tool)
# ---------------------------------------------------------------------------


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, mode: str, total: int, master_rows: int) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "running",
            "mode": mode,
            "created_at": _now(),
            "finished_at": None,
            "total": total,
            "completed": 0,
            "master_rows": master_rows,
            "thresholds": thresholds(),
            "records": [],
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def append_record(self, job_id: str, record: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["records"].append(record)
            job["completed"] = len(job["records"])

    def finish(self, job_id: str, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "error" if error else "done"
            job["error"] = error
            job["finished_at"] = _now()

    def update_record(self, job_id: str, record_id: str, patch: dict) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for rec in job["records"]:
                if rec.get("record_id") == record_id:
                    rec.update(patch)
                    return dict(rec)
            return None


STORE = JobStore()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

_RAW_INPUT_CANDIDATES = ("raw_input", "input", "vehicle", "text", "description")


def _read_master(upload: Optional[UploadFile]) -> pd.DataFrame:
    if upload is None:
        return load_mmv_master()
    raw = upload.file.read()
    df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    required = {"mmv_id", "make", "model", "variant"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            422,
            f"Master CSV is missing required column(s): {', '.join(sorted(missing))}. "
            "Expected mmv_id, make, model, variant, fuel_type, transmission, cc, "
            "seating_capacity, year_start, year_end.",
        )
    # Coerce numeric columns so the tools see ints, not strings.
    for col in ("cc", "seating_capacity", "year_start", "year_end"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def _read_inputs(upload: UploadFile) -> list[dict]:
    raw = upload.file.read()
    df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    if df.empty:
        raise HTTPException(422, "Input CSV has no rows.")
    raw_col = next((c for c in _RAW_INPUT_CANDIDATES if c in df.columns), None)
    if raw_col is None:
        # Fall back to the first column if nothing is named recognizably.
        raw_col = df.columns[0]
    has_id = "input_id" in df.columns

    records: list[dict] = []
    for i, row in df.iterrows():
        text = str(row[raw_col]).strip()
        if not text:
            continue
        rec = {
            "input_id": str(row["input_id"]).strip() if has_id and str(row["input_id"]).strip()
            else f"R{i + 1:02d}",
            "raw_input": text,
        }
        for col in df.columns:
            if col in (raw_col, "input_id"):
                continue
            val = str(row[col]).strip()
            if val:
                rec[col] = val
        records.append(rec)
    if not records:
        raise HTTPException(422, f"No usable rows found (looked at column '{raw_col}').")
    return records


# ---------------------------------------------------------------------------
# Job runners
# ---------------------------------------------------------------------------


def _run_mock(job_id: str, records: list[dict], master_df: pd.DataFrame) -> None:
    started = time.monotonic()
    log.info("job start mode=mock records=%d master_rows=%d", len(records), len(master_df))
    try:
        pipeline = MockPipeline(master_df)
        counts: dict[str, int] = {}
        for i, rec in enumerate(records, 1):
            rid = rec.get("input_id", "?")
            t0 = time.monotonic()
            state = pipeline.run(rec)
            decision = state.get("final_decision", "?")
            counts[decision] = counts.get(decision, 0) + 1
            log.info(
                "record %d/%d %s -> %s conf=%.2f (%dms)",
                i, len(records), rid, decision,
                float(state.get("confidence") or 0.0),
                int((time.monotonic() - t0) * 1000),
            )
            STORE.append_record(job_id, build_record_result(rec, state, "mock"))
        STORE.finish(job_id)
        log.info(
            "job done mode=mock records=%d breakdown=%s (%.1fs)",
            len(records), counts, time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("job failed mode=mock")
        STORE.finish(job_id, error=str(exc))


def _run_live(job_id: str, records: list[dict], master_df: pd.DataFrame) -> None:
    started = time.monotonic()
    log.info("job start mode=live records=%d master_rows=%d", len(records), len(master_df))
    try:
        from services.retrieval_service import RetrievalService
        from llm_touchpoints.normalization_llm import normalize_record
        from llm_touchpoints.query_reformulation_llm import reformulate_queries
        from graph import build_graph
        from services.matching_pipeline import MockPipeline  # for fail-safe explain

        retrieval = RetrievalService(df=master_df)
        graph_app = build_graph()
        log.info("live pipeline built (retrieval + graph); processing %d records", len(records))

        counts: dict[str, int] = {}
        errors = 0
        for i, rec in enumerate(records, 1):
            rid = rec.get("input_id", "?")
            t0 = time.monotonic()
            try:
                log.info("record %d/%d %s: %r", i, len(records), rid, rec.get("raw_input", ""))
                normalized = normalize_record({"raw_input": rec["raw_input"]})
                queries = reformulate_queries(normalized)
                candidates = _retrieve_multi(retrieval, queries, CANDIDATE_TOP_N)
                log.info("record %s retrieved %d candidates", rid, len(candidates))
                state = graph_app.invoke({
                    "input_record": rec,
                    "normalized_record": normalized,
                    "candidate_records": candidates,
                })
                state.setdefault("input_record", rec)
            except Exception as exc:  # noqa: BLE001 - fail safe per record
                errors += 1
                log.exception("record %s failed, flagging for review", rid)
                state = {
                    "input_record": rec,
                    "normalized_record": {"raw_input": rec["raw_input"]},
                    "search_queries": [],
                    "candidate_records": [],
                    "selected_match": {},
                    "reasoning_decision": {},
                    "reasoning_trace": [],
                    "validation_status": "not_validated",
                    "verifier_result": None,
                    "confidence": 0.0,
                    "explanation": (
                        f"Flagged for review: '{rec['raw_input']}' could not be "
                        f"processed automatically ({exc})."
                    ),
                    "final_decision": "review",
                    "step_count": 0,
                    "llm_call_log": [],
                    "error": str(exc),
                }
            decision = state.get("final_decision", "?")
            counts[decision] = counts.get(decision, 0) + 1
            log.info(
                "record %d/%d %s -> %s conf=%.2f (%dms)",
                i, len(records), rid, decision,
                float(state.get("confidence") or 0.0),
                int((time.monotonic() - t0) * 1000),
            )
            STORE.append_record(job_id, build_record_result(rec, state, "live"))
        STORE.finish(job_id)
        log.info(
            "job done mode=live records=%d breakdown=%s errors=%d (%.1fs)",
            len(records), counts, errors, time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("job failed mode=live")
        STORE.finish(job_id, error=str(exc))


def _retrieve_multi(service, queries: list, top_n: int) -> list:
    merged: dict = {}
    for query in queries:
        for cand in service.retrieve(query, top_n=top_n):
            mid = cand.get("mmv_id")
            existing = merged.get(mid)
            if existing is None or cand["combined_score"] > existing["combined_score"]:
                merged[mid] = cand
    ranked = sorted(merged.values(), key=lambda c: c["combined_score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    try:
        rows = len(load_mmv_master())
    except Exception:  # noqa: BLE001
        rows = 0
    return {
        "ok": True,
        "gemini_key_present": _has_api_key(),
        "model": os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        "thresholds": thresholds(),
        "default_master_rows": rows,
    }


@app.get("/api/sample")
def sample() -> dict:
    if not _SAMPLE_INPUTS.exists():
        raise HTTPException(404, "No bundled sample inputs found.")
    text = _SAMPLE_INPUTS.read_text(encoding="utf-8")
    return {"filename": _SAMPLE_INPUTS.name, "csv": text}


@app.post("/api/match")
def match(
    inputs: UploadFile = File(...),
    master: Optional[UploadFile] = File(None),
    mode: str = Form("mock"),
) -> JSONResponse:
    mode = mode if mode in ("mock", "live") else "mock"
    if mode == "live" and not _has_api_key():
        raise HTTPException(
            400,
            "Live mode needs a Gemini API key (GEMINI_API_KEY). Set one in .env, or "
            "use mock mode to run the deterministic pipeline without an LLM.",
        )

    master_df = _read_master(master)
    records = _read_inputs(inputs)

    job = STORE.create(mode=mode, total=len(records), master_rows=len(master_df))
    log.info(
        "match accepted job=%s mode=%s records=%d master_rows=%d",
        job["job_id"], mode, len(records), len(master_df),
    )
    runner = _run_mock if mode == "mock" else _run_live
    thread = threading.Thread(
        target=runner,
        args=(job["job_id"], records, master_df),
        name=f"job-{job['job_id']}",
        daemon=True,
    )
    thread.start()
    return JSONResponse({"job_id": job["job_id"], "mode": mode, "total": len(records)})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.post("/api/jobs/{job_id}/records/{record_id}/override")
def override(job_id: str, record_id: str, body: dict) -> dict:
    decision = str(body.get("decision", "")).lower()
    if decision not in ("approved", "review", "rejected"):
        raise HTTPException(422, "decision must be one of: approved, review, rejected.")
    patch = {
        "override": {
            "decision": decision,
            "note": str(body.get("note", "")).strip(),
            "at": _now(),
        }
    }
    updated = STORE.update_record(job_id, record_id, patch)
    if updated is None:
        raise HTTPException(404, "Unknown job or record id.")
    return updated


# Serve the built UI (ui/dist) when present, so `uvicorn server:app` can host the
# whole app in production. In dev, the Vite server proxies /api here instead.
_UI_DIST = _PROJECT_ROOT / "ui" / "dist"
if _UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
