"""state.py: the schema, its reducer, and the evidence gate."""

import operator

from langgraph.graph import StateGraph
from pydantic import ValidationError

from myassistant.state import (
    AssistantState,
    ConfidenceTier,
    Verdict,
    render_evidence,
    tier_from_observations,
)
from myassistant.tools.observation import failed, fetched

REAL_PAGE = "Kubernetes schedules containers across a cluster. " * 40


def _ok():
    return fetched(REAL_PAGE, source="https://example.com", status=200)


def _bad():
    return failed("HTTP 403", source="https://blocked.example", status=403)


def _channels():
    """What LangGraph actually resolves the schema to.

    Asserted here rather than on __annotations__, which `from __future__ import
    annotations` leaves as unresolved ForwardRefs - and because the resolved
    channel is the thing that governs how updates merge.
    """
    return StateGraph(AssistantState).channels


def test_observations_accumulate_rather_than_replace():
    """Without the reducer a later node's update silently erases earlier evidence."""
    assert _channels()["observations"].operator is operator.add


def test_scalar_keys_overwrite():
    """A tier or verdict is the latest answer, not a growing list."""
    for key in ("confidence", "self_report", "verdict", "revisions"):
        assert type(_channels()[key]).__name__ == "LastValue"


def test_messages_keeps_the_supervisor_shape():
    """langgraph-supervisor routes on messages, so it must stay an aggregate."""
    assert type(_channels()["messages"]).__name__ == "BinaryOperatorAggregate"


def test_every_key_is_optional():
    """Nodes return partial updates - a node that only adds evidence returns
    {"observations": [...]} and LangGraph merges it. Requiring any key would
    make that a type error for no runtime benefit."""
    assert AssistantState.__required_keys__ == frozenset()
    assert "observations" in AssistantState.__optional_keys__


def test_one_good_observation_is_enough_for_high():
    assert tier_from_observations([_bad(), _ok()]) is ConfidenceTier.HIGH


def test_all_failed_observations_means_low():
    assert tier_from_observations([_bad(), _bad()]) is ConfidenceTier.LOW


def test_no_observations_means_low_not_high():
    """Calling no tools is not evidence of anything - it must not read as HIGH."""
    assert tier_from_observations([]) is ConfidenceTier.LOW


def test_gate_reads_ok_not_the_presence_of_text():
    """A block page carries plenty of text; the gate must still call it LOW."""
    block_page = fetched("Please enable JavaScript. " * 20, source="https://x.example")
    assert block_page.content == ""
    assert tier_from_observations([block_page]) is ConfidenceTier.LOW


def test_tiers_are_labels_not_numbers():
    """Tiers exist so a percentage never gets invented."""
    assert ConfidenceTier.HIGH == "high"
    assert ConfidenceTier.UNGROUNDED == "ungrounded"


def test_critic_sees_the_evidence_including_the_failures():
    rendered = render_evidence([_ok(), _bad()])
    assert "[OK]" in rendered
    assert "[TOOL FAILED]" in rendered
    assert "status=403" in rendered


def test_no_tools_called_is_said_explicitly():
    """An empty string would read to the critic as an absent field, not as 'none'."""
    assert render_evidence([]) == "(no tools were called)"


def test_verdict_requires_a_decision():
    v = Verdict(supported=False, issue="cites a page that returned 403")
    assert v.supported is False
    assert v.issue == "cites a page that returned 403"


def test_verdict_rejects_a_missing_decision():
    """Structured output means the model cannot answer with prose instead."""
    try:
        Verdict(issue="unsure")
    except ValidationError:
        pass
    else:
        raise AssertionError("supported must be required")
