"""`/stats` - what the traces say, in the terminal.

Langfuse is the system of record; this is a second way to look at it. The
browser dashboard is better for anything deep, but a question you would
otherwise not bother opening a browser for is a question you stop asking, and
the whole point of wiring tracing in from step 0 was to answer those.

Reads, never writes. It cannot change what happened, only report it - so every
failure path here degrades to a printed line rather than an exception.

What it reports, and why each one:
  turns, and how long        the responsiveness priority, measured not assumed
  the slowest turns          an average hides the 204-second outlier
  routing                    which agent ran, and how often
  confidence tiers           how often an answer was actually grounded
  models                     where the time goes, per model
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from myassistant import config

log = logging.getLogger("myassistant")

# Enough to see a pattern without waiting on a slow query. Langfuse paginates;
# this is one page.
DEFAULT_LIMIT = 100

# A turn slower than this is worth naming individually rather than averaging
# away. Set from observation: research turns land near 30s, a coder turn with a
# file read reached 204s.
SLOW_S = 45.0

# How many recent traces to pull full detail for. Scores need one request each,
# so this is the trade between a useful distribution and a command that returns
# promptly.
SCORE_SAMPLE = 15


def _client() -> Any:
    """The Langfuse client, or None if tracing is not configured or reachable."""
    if not config.LANGFUSE_ENABLED:
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        return client if client.auth_check() else None
    except Exception:  # a stopped container is not an error here
        log.exception("could not reach Langfuse")
        return None


def _bar(count: int, total: int, width: int = 24) -> str:
    """A proportion, drawn. Easier to compare at a glance than percentages."""
    filled = round(width * count / total) if total else 0
    return "█" * filled + "·" * (width - filled)


def _agent_of(trace: Any) -> str:
    """Which agent handled a turn, read from its output text.

    Read from the output rather than from a span name because the agent's own
    message carries its name, and that survives however the graph is wired.
    """
    text = str(getattr(trace, "output", "") or "")
    for name in ("coding_agent", "docs_agent", "research_agent", "general_agent"):
        if name in text:
            return name
    return "(unrouted)"


def _tiers(client: Any, traces: list[Any]) -> Counter[str]:
    """Confidence tiers, from a sample of recent turns.

    A sample, not everything: the list endpoint returns score *ids* only, and
    the full objects need one request per trace. Reading fifty would make a
    REPL command feel broken, and the distribution of the last handful answers
    the question anyway - how often is an answer actually grounded.
    """
    tiers: Counter[str] = Counter()
    for trace in traces:
        if not getattr(trace, "scores", None):
            continue
        try:
            detail = client.fetch_trace(trace.id).data
        except Exception:  # one unreadable trace must not lose the report
            log.exception("could not fetch trace %s", trace.id)
            continue
        for score in getattr(detail, "scores", None) or []:
            if getattr(score, "name", "") == "confidence":
                value = getattr(score, "string_value", None) or getattr(score, "value", None)
                if value is not None:
                    tiers[str(value)] += 1
    return tiers


def parse_window(arg: str) -> tuple[str | None, str]:
    """Turn a /stats argument into (period, label). "" means everything.

    Accepts "24h", "3d", "30m" and "all". Anything else is treated as "all"
    rather than refused: a report is not worth an argument about syntax.
    """
    arg = arg.strip().lower()
    if not arg or arg == "all":
        return None, "all time"
    units = {"m": ("minutes", "minute"), "h": ("hours", "hour"), "d": ("days", "day")}
    if arg[-1] in units and arg[:-1].isdigit():
        amount, (unit, singular) = int(arg[:-1]), units[arg[-1]]
        return arg, f"last {amount} {singular if amount == 1 else unit}"
    return None, "all time"


def _since(window: str | None) -> Any:
    """The from_timestamp for a window like "24h", or None."""
    if not window:
        return None
    from datetime import UTC, datetime, timedelta

    unit = {"m": "minutes", "h": "hours", "d": "days"}[window[-1]]
    return datetime.now(UTC) - timedelta(**{unit: int(window[:-1])})


def summary(
    limit: int = DEFAULT_LIMIT,
    *,
    session_id: str | None = None,
    window: str | None = None,
    scope: str = "all time",
) -> str:
    """The text `/stats` prints. Never raises - a report cannot be worth a crash.

    Scoping matters more than it looks. Traces accumulate across every session
    and every experiment, so an unfiltered count answers "what has this
    project ever done" rather than "how is it behaving now" - and the second is
    the question worth asking after a change.
    """
    client = _client()
    if client is None:
        return (
            "no tracing data: Langfuse is not configured or not running\n"
            "  docker compose -f docker-compose.langfuse.yml up -d"
        )

    query: dict[str, Any] = {"limit": limit}
    if session_id:
        query["session_id"] = session_id
    if (start := _since(window)) is not None:
        query["from_timestamp"] = start

    try:
        traces = client.fetch_traces(**query).data
    except Exception as e:
        log.exception("could not fetch traces")
        return f"could not read traces: {e}"

    if not traces:
        return f"no traces for {scope}" + (
            " - ask a question first" if session_id else " - try /stats all"
        )

    latencies = sorted(
        (t.latency, str(getattr(t, "name", "?"))) for t in traces if getattr(t, "latency", None)
    )
    values = [seconds for seconds, _ in latencies]
    lines = [f"{len(traces)} turns · {scope} · {config.LANGFUSE_HOST}"]

    if values:
        median = values[len(values) // 2]
        lines.append(
            f"  latency   median {median:.1f}s   fastest {values[0]:.1f}s   "
            f"slowest {values[-1]:.1f}s"
        )
        slow = [s for s in values if s >= SLOW_S]
        if slow:
            # Named rather than averaged: the outlier is the interesting part,
            # and a mean would hide it behind a pile of 2-second replies.
            lines.append(f"  {len(slow)} turns over {SLOW_S:.0f}s, worst {values[-1]:.0f}s")

    routed = Counter(_agent_of(t) for t in traces)
    if routed:
        lines.append("")
        lines.append("  routing")
        for name, count in routed.most_common():
            lines.append(f"    {name:16} {_bar(count, len(traces))} {count}")

    tiers = _tiers(client, traces[:SCORE_SAMPLE])
    if tiers:
        lines.append("")
        lines.append(f"  confidence (last {sum(tiers.values())} scored)")
        total = sum(tiers.values())
        for value, count in tiers.most_common():
            lines.append(f"    {value:16} {_bar(count, total)} {count}")

    return "\n".join(lines)
