"""stats.py: a report can never be worth a crash."""

from types import SimpleNamespace

import pytest

from myassistant import config
from myassistant.observability import stats


def _trace(latency=1.0, output="", scores=None, tid="t"):
    return SimpleNamespace(
        id=tid, latency=latency, name="LangGraph", output=output, scores=scores or []
    )


class _Client:
    def __init__(self, traces, detail=None, raises=False):
        self._traces, self._detail, self._raises = traces, detail, raises

    def fetch_traces(self, limit=100, **query):
        if self._raises:
            raise ConnectionError("langfuse is down")
        self.query = query
        return SimpleNamespace(data=self._traces[:limit])

    def fetch_trace(self, trace_id):
        return SimpleNamespace(data=self._detail)


@pytest.fixture
def client(monkeypatch):
    def _install(traces, detail=None, raises=False):
        c = _Client(traces, detail, raises)
        monkeypatch.setattr(stats, "_client", lambda: c)
        return c

    return _install


# --- the failure paths, which matter most here ---


def test_langfuse_not_configured_says_how_to_start_it(monkeypatch):
    """A report that just says "nothing" leaves you with nothing to do."""
    monkeypatch.setattr(stats, "_client", lambda: None)
    out = stats.summary()
    assert "not configured or not running" in out
    assert "docker compose" in out


def test_a_failed_fetch_returns_a_line_not_an_exception(client):
    client([], raises=True)
    assert "could not read traces" in stats.summary()


def test_no_traces_is_not_an_error(client):
    client([])
    assert "no traces for" in stats.summary()


def test_one_unreadable_trace_does_not_lose_the_report(client, monkeypatch):
    c = client([_trace(scores=["s1"])])
    monkeypatch.setattr(
        c, "fetch_trace", lambda tid: (_ for _ in ()).throw(ConnectionError("gone"))
    )
    assert "1 turns" in stats.summary()


# --- what it reports ---


def test_latency_reports_the_median_not_the_mean(client):
    """A mean would be dragged around by one 204-second outlier."""
    client([_trace(latency=x) for x in (1, 2, 3, 4, 200)])
    assert "median 3.0s" in stats.summary()


def test_slow_turns_are_named_rather_than_averaged_away(client):
    """The outlier is the interesting part - it must not hide behind an average."""
    client([_trace(latency=x) for x in (1, 1, 1, 204)])
    out = stats.summary()
    assert "1 turns over 45s" in out
    assert "worst 204s" in out


def test_no_slow_line_when_everything_is_fast(client):
    client([_trace(latency=x) for x in (1, 2, 3)])
    assert "turns over" not in stats.summary()


def test_routing_is_counted_per_agent(client):
    client(
        [
            _trace(output="from research_agent"),
            _trace(output="from docs_agent"),
            _trace(output="from research_agent"),
        ]
    )
    out = stats.summary()
    assert "research_agent" in out
    assert "docs_agent" in out


def test_a_turn_that_never_reached_an_agent_is_visible(client):
    """The supervisor sometimes answers nothing and hands off to no one. That
    is a real failure mode and should show up, not be silently dropped."""
    client([_trace(output="")])
    assert "(unrouted)" in stats.summary()


def test_confidence_tiers_are_counted(client):
    score = SimpleNamespace(name="confidence", string_value="high", value=None)
    client([_trace(scores=["s1"])], detail=SimpleNamespace(scores=[score]))
    out = stats.summary()
    assert "confidence" in out
    assert "high" in out


def test_scores_other_than_confidence_are_ignored(client):
    """self_reported_confidence is logged deliberately and must not be shown -
    it is a calibration experiment, not something to present as a tier."""
    other = SimpleNamespace(name="self_reported_confidence", string_value=None, value=0.9)
    client([_trace(scores=["s1"])], detail=SimpleNamespace(scores=[other]))
    assert "0.9" not in stats.summary()


def test_the_score_sample_is_bounded(client):
    """One request per trace - reading fifty would make /stats feel broken."""
    assert stats.SCORE_SAMPLE < stats.DEFAULT_LIMIT


def test_the_host_is_shown_so_the_dashboard_can_be_opened(client):
    client([_trace()])
    assert config.LANGFUSE_HOST in stats.summary()


# --- scoping ---


def test_a_window_argument_is_parsed():
    assert stats.parse_window("24h") == ("24h", "last 24 hours")
    assert stats.parse_window("3d") == ("3d", "last 3 days")
    assert stats.parse_window("1d") == ("1d", "last 1 day")  # singular
    assert stats.parse_window("30m") == ("30m", "last 30 minutes")


def test_all_and_empty_mean_everything():
    assert stats.parse_window("") == (None, "all time")
    assert stats.parse_window("all") == (None, "all time")


def test_nonsense_falls_back_rather_than_refusing():
    """A report is not worth an argument about syntax."""
    assert stats.parse_window("banana") == (None, "all time")
    assert stats.parse_window("24x") == (None, "all time")


def test_a_session_filter_is_passed_to_langfuse(client):
    c = client([_trace()])
    stats.summary(session_id="abc", scope="this session")
    assert c.query["session_id"] == "abc"


def test_a_window_becomes_a_from_timestamp(client):
    c = client([_trace()])
    stats.summary(window="24h", scope="last 24 hours")
    assert "from_timestamp" in c.query


def test_no_filter_sends_no_narrowing_query(client):
    c = client([_trace()])
    stats.summary(scope="all time")
    assert "session_id" not in c.query
    assert "from_timestamp" not in c.query


def test_the_scope_is_stated_in_the_output(client):
    """46 turns means nothing without saying 46 turns of what."""
    client([_trace()])
    assert "this session" in stats.summary(scope="this session")


def test_an_empty_session_suggests_asking_something(client):
    client([])
    assert "ask a question first" in stats.summary(session_id="abc", scope="this session")


def test_an_empty_window_suggests_widening_it(client):
    """Different advice: there is history, just not in this window."""
    client([])
    assert "/stats all" in stats.summary(window="24h", scope="last 24 hours")
