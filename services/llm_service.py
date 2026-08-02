"""Shared LLM client wrapper for all six touchpoints.

Centralizes model config, message construction, retries, and cost/latency
capture (fed into ``llm_call_log``). Every touchpoint calls through here rather
than instantiating its own client.

Provider: Google Gemini via the ``google-genai`` SDK. The API key is read from
the environment (``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``) so no
secret is ever hard-coded. The model id defaults to ``gemini-2.5-flash`` and can
be overridden with ``GEMINI_MODEL``.

Structured output: callers use :meth:`LLMService.generate_json`, which forces
``response_mime_type="application/json"`` and parses the reply into a Python
dict. An optional JSON schema can be passed to constrain the shape.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Load key/model config from the project-root .env if present, so callers don't
# have to export vars manually. Values already in the environment win over .env.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv optional; env vars still work without it.
    pass

# --- Defaults -------------------------------------------------------------

# Use the rolling "flash-latest" alias rather than a pinned version: pinned
# releases get retired (e.g. gemini-2.5-flash started 404ing for new keys), and
# the alias always resolves to the current flash. Override with GEMINI_MODEL.
_DEFAULT_MODEL = "gemini-flash-latest"
_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 1.0  # seconds; exponential backoff between retries.
# Never wait longer than this for a single retry, even if the server asks for
# more — keeps a stuck free-tier daily quota from hanging the whole run.
_MAX_RETRY_DELAY = 30.0


def _retry_delay_from_error(exc: Exception) -> Optional[float]:
    """Extract the server-suggested retryDelay (seconds) from a 429, if any.

    Gemini 429s carry a RetryInfo detail like ``{'retryDelay': '20s'}``; honoring
    it is far more effective against per-minute rate limits than blind backoff.
    """
    details = getattr(exc, "details", None)
    if not details:
        return None
    error = details.get("error", details) if isinstance(details, dict) else {}
    for detail in error.get("details", []) if isinstance(error, dict) else []:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).strip().rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _read_api_key() -> str:
    for var in _API_KEY_ENV_VARS:
        key = os.environ.get(var)
        if key:
            return key
    raise RuntimeError(
        "No Gemini API key found. Set one of: "
        + ", ".join(_API_KEY_ENV_VARS)
        + " in your environment."
    )


@dataclass
class LLMCall:
    """One record of an LLM invocation, appended to ``llm_call_log``."""

    touchpoint: str
    model: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    ok: bool = True
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "touchpoint": self.touchpoint,
            "model": self.model,
            "latency_s": round(self.latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "ok": self.ok,
            "error": self.error,
            **self.extra,
        }


class LLMService:
    """Single entrypoint for LLM calls across the pipeline."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key or _read_api_key())
        # Every call made through this service is recorded here; the graph can
        # copy it into MMVState["llm_call_log"] for eval.
        self.call_log: list[dict] = []

    def generate_json(
        self,
        touchpoint: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Any] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Call the model and return its reply parsed as a JSON object.

        ``touchpoint`` labels the call in ``call_log``. ``response_schema`` is an
        optional ``google.genai`` schema (or Pydantic type) used to constrain the
        output shape; when omitted we still force JSON mime type and rely on the
        prompt to specify the structure.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=self.temperature if temperature is None else temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        text, call = self._generate(touchpoint, user_prompt, config)
        self.call_log.append(call.to_dict())
        if not call.ok:
            raise RuntimeError(f"LLM call failed for {touchpoint}: {call.error}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{touchpoint}: model did not return valid JSON:\n{text}"
            ) from exc
        return data

    # --- internals --------------------------------------------------------

    def _generate(
        self,
        touchpoint: str,
        user_prompt: str,
        config: "types.GenerateContentConfig",
    ) -> tuple[str, LLMCall]:
        """Invoke the model with retries; return (text, call-record)."""
        last_error: Optional[Exception] = None
        start = time.perf_counter()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                latency = time.perf_counter() - start
                usage = getattr(response, "usage_metadata", None)
                call = LLMCall(
                    touchpoint=touchpoint,
                    model=self.model,
                    latency_s=latency,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                    ok=True,
                    extra={"attempts": attempt},
                )
                return (response.text or "", call)
            except genai_errors.APIError as exc:
                last_error = exc
                # Retry on transient (5xx / 429) errors; fail fast otherwise.
                status = getattr(exc, "code", None)
                if status not in (429, 500, 502, 503, 504) or attempt == _MAX_RETRIES:
                    break
                # Prefer the server's suggested delay (429s), else exponential
                # backoff; cap either so a daily-quota block can't hang forever.
                suggested = _retry_delay_from_error(exc)
                delay = suggested if suggested is not None else (
                    _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                )
                time.sleep(min(delay, _MAX_RETRY_DELAY))
            except Exception as exc:  # noqa: BLE001 - record and stop
                last_error = exc
                break

        latency = time.perf_counter() - start
        call = LLMCall(
            touchpoint=touchpoint,
            model=self.model,
            latency_s=latency,
            ok=False,
            error=str(last_error),
        )
        return ("", call)
