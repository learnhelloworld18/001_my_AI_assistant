"""The typed state passed between supervisor and agents.

One schema for the whole graph, so a node cannot invent a key or quietly change
a type - the hard rule against passing bare dicts around LangGraph.

Two different typing tools on purpose:
  AssistantState  a TypedDict, because that is what LangGraph merges between
                  nodes and what langgraph-supervisor expects
  Verdict         a Pydantic model, because it is a *model's* structured
                  output and needs real validation and a JSON schema
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from myassistant.tools.observation import Observation


class ConfidenceTier(StrEnum):
    """How much the evidence actually supports the answer.

    Tiers, not percentages: there is no calibrated probability model here, only
    a few concrete signals, and "82% confident" would be false precision.
    """

    HIGH = "high"  # tools returned real content / validation passed
    LOW = "low"  # thin or failed evidence - must be said out loud in the answer
    UNGROUNDED = "ungrounded"  # general_agent: no tools, so nothing to check against


class Verdict(BaseModel):
    """The critic's answer. Binary plus a reason, never a score.

    Pydantic rather than a dataclass because this is parsed out of a model's
    output - the validation is the point, and it doubles as the JSON schema
    handed to the model.
    """

    supported: bool = Field(description="Is the answer supported by the tool evidence?")
    issue: str | None = Field(default=None, description="If not, what is unsupported")


# total=False so nodes can return partial updates - LangGraph merges them, and
# every key below is absent until some node has actually produced it.
#
# Spelled out rather than subclassing MessagesState: langgraph ships no type
# stubs, so mypy sees that base as Any and cannot check anything declared on
# top of it. Same resolved channels either way (asserted in tests/test_state.py).
class AssistantState(TypedDict, total=False):
    """What flows between the supervisor, the agents and the critic."""

    # add_messages is the reducer langgraph-supervisor routes on: it appends,
    # and reconciles ids rather than duplicating a message on a re-emit.
    messages: Annotated[list[AnyMessage], add_messages]

    # operator.add makes this accumulate too. Without a reducer a node returning
    # observations would *replace* the list, so evidence from an earlier tool
    # call would vanish exactly when the gate needs to weigh it.
    observations: Annotated[list[Observation], operator.add]

    confidence: ConfidenceTier  # evidence-based, and the only tier ever shown
    self_report: float  # the model's own number - logged, never displayed
    verdict: Verdict  # set only when CRITIC_ENABLED
    revisions: int  # critic retries used, against CRITIC_MAX_REVISIONS


def tier_from_observations(observations: list[Observation]) -> ConfidenceTier:
    """Default evidence gate: did any tool actually return usable content?

    This is research_agent's rule. docs_agent (RAG score threshold) and
    coding_agent (validate_code passed) read their own signals out of
    Observation.metrics instead - same evidence, different question.

    Deliberately reads Observation.ok rather than asking the model whether it
    succeeded: the model can only notice a failure that is visible in the
    observation text, and this exists to catch the ones that are not.
    """
    return ConfidenceTier.HIGH if any(o.ok for o in observations) else ConfidenceTier.LOW


def render_evidence(observations: list[Observation]) -> str:
    """The evidence block handed to the critic.

    The critic judges "is this answer supported by what the tools returned", so
    it gets the observations themselves. Given only the prose it would be a
    model grading a model against nothing.
    """
    if not observations:
        return "(no tools were called)"
    return "\n".join(o.render() for o in observations)
