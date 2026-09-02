"""The observation contract every tool returns.

The agent loop has no separate evaluate step - "evaluate" is just the model
reasoning again with the observation now in context. So the model's ability to
notice a problem is bounded by what this text makes visible: a tool that fails
quietly and returns something plausible leaves nothing to notice, and the step
cap never saves you, because an agent that believes it succeeded stops looping.

Two consumers, one object:
  render()      the text the model sees - failure stated first, loudly
  ok / metrics  what the graph's evidence gate reads, which never depends on
                the model having noticed anything
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Below this, a "successful" fetch is a consent wall, a JS shell or an error
# page - superficially fine text that reads as a real page to the model.
# Turning that into an explicit failure is the whole point of this module.
MIN_USEFUL_CHARS = 400

# Substrings that mean "this is a block page, not the content you asked for".
# Only checked near the start, where such notices actually appear - a page
# legitimately *about* captchas shouldn't be thrown away for saying the word.
_NOT_REAL_CONTENT = (
    "enable javascript",
    "access denied",
    "are you a robot",
    "verify you are human",
    "captcha",
)
_MARKER_SCAN_CHARS = 2000


@dataclass(frozen=True)
class Observation:
    """One tool result. Never return a bare string from a tool instead of this.

    frozen=True because an observation is a record of something that already
    happened - a later node rewriting it would be falsifying the evidence the
    confidence tier is derived from.
    """

    ok: bool
    detail: str  # what happened, in words - populated on success too
    content: str = ""  # the payload, empty on failure
    source: str | None = None  # url/path, so the model can try a different one
    metrics: dict[str, Any] = field(default_factory=dict)  # chars, status, score

    def render(self) -> str:
        """The text handed back to the model.

        Failures lead with a marker so they can't be skimmed past, and every
        result carries its numbers - the model can only notice that a result
        was thin if it can see that it was thin.
        """
        head = "OK" if self.ok else "TOOL FAILED"
        parts = [f"[{head}] {self.detail}"]
        if self.metrics:
            parts.append("(" + " ".join(f"{k}={v}" for k, v in self.metrics.items()) + ")")
        if self.source:
            parts.append(f"source: {self.source}")
        line = " ".join(parts)
        return f"{line}\n{self.content}" if self.content else line


def looks_empty(text: str) -> str | None:
    """Return why this content isn't usable, or None if it looks real.

    Deliberately hand-rolled and crude: a few string checks catch the common
    cases, and a wrong guess costs one retry, not a wrong answer.
    """
    stripped = text.strip()
    if len(stripped) < MIN_USEFUL_CHARS:
        return f"only {len(stripped)} chars extracted - not a real page body"

    head = stripped[:_MARKER_SCAN_CHARS].lower()
    for marker in _NOT_REAL_CONTENT:
        if marker in head:
            return f"page body looks like a block/consent page ({marker!r})"
    return None


def fetched(content: str, source: str, **metrics: Any) -> Observation:
    """Build an Observation for content a tool believes it fetched successfully.

    Routes every "success" through looks_empty() so a 200-with-a-block-page
    becomes a loud failure rather than a plausible-looking answer. Tools should
    use this rather than constructing Observation(ok=True) by hand.
    """
    problem = looks_empty(content)
    chars = len(content.strip())
    if problem is not None:
        return Observation(
            ok=False,
            detail=f"fetch returned no usable content: {problem}",
            source=source,
            metrics={"chars": chars, **metrics},
        )
    return Observation(
        ok=True,
        detail="extracted page content",
        content=content,
        source=source,
        metrics={"chars": chars, **metrics},
    )


def emit(obs: Observation, tool_call_id: str) -> Command:
    """Send one Observation to both of its consumers.

    A LangChain tool's return becomes a ToolMessage whose content is a string,
    so the object would otherwise be flattened away the moment it is returned.
    A Command carries both: the rendered text for the model to reason over, and
    the Observation itself for the in-graph evidence gate, which must not have
    to parse prose to find out whether a tool worked.
    """
    return Command(
        update={
            "observations": [obs],
            "messages": [ToolMessage(content=obs.render(), tool_call_id=tool_call_id)],
        }
    )


def failed(detail: str, source: str | None = None, **metrics: Any) -> Observation:
    """Build an explicit failure. Use for exceptions, bad status codes, no results.

    Exists so no tool is ever tempted to return "" or "No results found" - both
    read to the model as a legitimate empty answer rather than a problem.
    """
    return Observation(ok=False, detail=detail, source=source, metrics=metrics)
