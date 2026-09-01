"""Langfuse tracing: one handler for the whole graph, or nothing at all.

Langfuse is the flight recorder - it watches, it never drives. Attach the
handler to the supervisor graph once and every model call, tool call and
routing decision shows up as a span under one trace at localhost:3000, with
Session.session_id grouping a run's turns together.

The property that matters is that it is *optional*. Langfuse runs in Docker,
and Docker is often not running. A local assistant must not refuse to answer
because its logging is down, so every path here degrades to "no tracing" and
the caller cannot tell the difference.

Import this after myassistant.config - that is where load_dotenv() runs.
"""

from __future__ import annotations

import logging
from typing import Any

from myassistant import config

log = logging.getLogger("myassistant")

# Network calls here are on the REPL's critical path, so they get a short leash.
_TIMEOUT_S = 5

# Guarded because the Langfuse v2 SDK imports langchain.callbacks, which
# LangChain 1.x removed. pyproject pins 0.3.x, but a stray upgrade should cost
# tracing, not the whole REPL.
try:
    from langfuse.callback import CallbackHandler

    _IMPORT_ERROR: str | None = None
except ImportError as e:  # pragma: no cover - only on a broken install
    CallbackHandler = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = str(e)


# Built once per process, then reused: auth_check() is a network round trip and
# whether Langfuse is up cannot change usefully mid-run. _built separates "not
# tried yet" from "tried, unavailable" - both leave _handler as None.
_handler: Any | None = None
_built = False


def _build(session_id: str) -> Any | None:
    """Create the handler, or return None if tracing isn't available."""
    if not config.LANGFUSE_ENABLED:
        log.info("langfuse: no keys configured, running without tracing")
        return None

    if CallbackHandler is None:
        log.warning("langfuse: import failed (%s), running without tracing", _IMPORT_ERROR)
        return None

    handler = CallbackHandler(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
        session_id=session_id,
        timeout=_TIMEOUT_S,
    )

    # Checked out loud, once. The SDK swallows its own errors by design, so a
    # stopped container otherwise means silently zero traces - the same
    # looks-fine-but-isn't failure that tools/observation.py exists to prevent.
    # auth_check() raises ConnectError when the host is down, so bad keys and an
    # unreachable host both have to be caught here.
    try:
        ok = handler.auth_check()
    except Exception as e:  # noqa: BLE001 - any failure must mean "no tracing", never a crash
        log.warning("langfuse: unreachable at %s (%s)", config.LANGFUSE_HOST, e)
        print("(langfuse unreachable - running without tracing)")
        return None

    if not ok:
        log.warning("langfuse: auth_check rejected the keys for %s", config.LANGFUSE_HOST)
        print("(langfuse rejected the API keys - running without tracing)")
        return None

    log.info("langfuse: tracing to %s, session %s", config.LANGFUSE_HOST, session_id)
    return handler


def get_callbacks(session_id: str) -> list[Any]:
    """Callbacks to hand LangGraph: [handler] when tracing, [] when not.

    The empty list *is* the no-op. LangChain accepts it happily, so no call
    site needs an `if tracing:` branch and there is no stub class to keep in
    sync with the real handler's interface.
    """
    global _handler, _built
    if not _built:
        _built = True
        _handler = _build(session_id)
    return [_handler] if _handler is not None else []


# Score names, kept deliberately apart. Merging them would launder a guess into
# the evidence channel, and comparing the two is the entire point.
CONFIDENCE_SCORE = "confidence"  # evidence-based, and the only one ever shown
SELF_REPORT_SCORE = "self_reported_confidence"  # logged for study, never shown


def score(name: str, value: str | float, comment: str | None = None) -> None:
    """Attach a score to the trace just produced - confidence tier, critic verdict.

    Anything scored under CONFIDENCE_SCORE is an evidence-based signal from
    PROJECT_REQUIREMENTS.md (did visit_webpage return real content, did
    validate_code pass) - never a model's self-reported number, which goes to
    score_self_report() instead. No-ops when tracing is off.
    """
    if _handler is None:
        return
    try:
        trace_id = _handler.get_trace_id()
        if trace_id:
            _handler.langfuse.score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception:  # recording a tier is never worth failing the user's question over
        log.exception("langfuse: could not record score %s", name)


def score_self_report(value: float, comment: str | None = None) -> None:
    """Log the model's own confidence number, for calibration study only.

    Never printed. A self-reported "87%" shown as a confidence tag reads as far
    more rigorous than it is - that is what the hard rule forbids, and the rule
    is about what the user sees. Logged under its own score name so it can be
    compared against the evidence tier rather than confused with it.

    The hypothesis this exists to test: verbalized confidence from LLMs is
    poorly calibrated and skewed high, and 3B local models are the worst case.
    Expect 0.85-0.95 almost always, including on turns where visit_webpage
    failed and the evidence tier says low. The interesting Langfuse query is
    the disagreement - high self-report against a low evidence tier.

    Off unless SELF_REPORT_ENABLED: asking for the number constrains the
    agent's generation, so it is an experiment you switch on, not a standing tax.
    """
    if not config.SELF_REPORT_ENABLED:
        return
    score(SELF_REPORT_SCORE, _as_fraction(value), comment)


def _as_fraction(value: float) -> float:
    """Normalise a confidence to 0-1 so the logged numbers are comparable.

    Small models ignore the output format and answer "90" where 0.9 was asked
    for, often in the same run - left raw, the series is unanalysable. Anything
    still out of range after that is clamped and logged rather than dropped.
    """
    if 1 < value <= 100:
        value = value / 100
    if not 0 <= value <= 1:
        log.warning("langfuse: self-report %s out of range, clamping", value)
        value = min(max(value, 0.0), 1.0)
    return value


def flush() -> None:
    """Send anything still buffered. Call on the way out of the REPL.

    Langfuse batches spans on a background thread, so a short-lived CLI can
    exit with the last turn never sent - silent loss of exactly the trace you
    just went looking for.
    """
    if _handler is None:
        return
    try:
        _handler.flush()
    except Exception:  # shutdown must not fail on logging
        log.exception("langfuse: flush failed, some spans were not sent")


def reset() -> None:
    """Drop the cached handler so the next call rebuilds it. Tests only."""
    global _handler, _built
    _handler, _built = None, False
