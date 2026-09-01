"""langfuse_client.py: the point is that tracing is never load-bearing.

Every test here is a way Langfuse can be broken; none of them may raise.
"""

import pytest

from myassistant.observability import langfuse_client as lc


@pytest.fixture(autouse=True)
def _fresh():
    """The handler is cached per process, so each test needs a clean slate."""
    lc.reset()
    yield
    lc.reset()


class _FakeHandler:
    """Stands in for CallbackHandler. Records what was asked of it."""

    def __init__(self, *, auth=True, trace_id="t1", **kwargs):
        self.kwargs = kwargs
        self._auth = auth
        self._trace_id = trace_id
        self.flushed = False
        self.scores = []
        self.langfuse = self

    def auth_check(self):
        if isinstance(self._auth, Exception):
            raise self._auth
        return self._auth

    def get_trace_id(self):
        return self._trace_id

    def score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed = True


def _install(monkeypatch, **handler_kwargs):
    """Enable tracing with a fake handler, returning the instance once built."""
    made = []

    def factory(**kwargs):
        h = _FakeHandler(**handler_kwargs, **kwargs)
        made.append(h)
        return h

    monkeypatch.setattr(lc.config, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(lc.config, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(lc.config, "LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(lc, "CallbackHandler", factory)
    return made


def test_no_keys_means_no_callbacks(monkeypatch):
    """The default state on a fresh clone must still run."""
    monkeypatch.setattr(lc.config, "LANGFUSE_ENABLED", False)
    assert lc.get_callbacks("s1") == []


def test_unreachable_host_degrades_instead_of_raising(monkeypatch):
    """auth_check raises ConnectError when the container is down - the common case."""
    _install(monkeypatch, auth=ConnectionError("connection refused"))
    assert lc.get_callbacks("s1") == []


def test_rejected_keys_degrade_instead_of_raising(monkeypatch):
    """Stale keys return False rather than raising. Same outcome required."""
    _install(monkeypatch, auth=False)
    assert lc.get_callbacks("s1") == []


def test_working_langfuse_returns_one_handler(monkeypatch):
    made = _install(monkeypatch)
    callbacks = lc.get_callbacks("s1")
    assert callbacks == [made[0]]
    assert made[0].kwargs["session_id"] == "s1"  # groups the run's traces


def test_handler_is_built_once(monkeypatch):
    """auth_check is a network round trip - paying it per turn is a latency bug."""
    made = _install(monkeypatch)
    for _ in range(5):
        lc.get_callbacks("s1")
    assert len(made) == 1


def test_failure_is_not_retried_every_turn(monkeypatch):
    """A down container must not cost a failed connection attempt on every turn."""
    made = _install(monkeypatch, auth=ConnectionError("refused"))
    for _ in range(5):
        assert lc.get_callbacks("s1") == []
    assert len(made) == 1


def test_score_records_against_the_current_trace(monkeypatch):
    made = _install(monkeypatch)
    lc.get_callbacks("s1")
    lc.score("confidence", "high", comment="visit_webpage returned 4kb")
    assert made[0].scores == [
        {
            "trace_id": "t1",
            "name": "confidence",
            "value": "high",
            "comment": "visit_webpage returned 4kb",
        }
    ]


def test_score_without_tracing_is_a_no_op(monkeypatch):
    """Agents call score() unconditionally; it must not need an `if` around it."""
    monkeypatch.setattr(lc.config, "LANGFUSE_ENABLED", False)
    lc.get_callbacks("s1")
    lc.score("confidence", "low")  # must not raise


def test_a_broken_score_does_not_break_the_turn(monkeypatch):
    """Recording a tier is never worth failing a user's question over."""
    made = _install(monkeypatch)
    lc.get_callbacks("s1")
    monkeypatch.setattr(made[0], "get_trace_id", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    lc.score("confidence", "high")  # must not raise


def test_flush_is_called_on_the_way_out(monkeypatch):
    """Without this the last turn's spans are lost - silently."""
    made = _install(monkeypatch)
    lc.get_callbacks("s1")
    lc.flush()
    assert made[0].flushed


def test_flush_without_tracing_is_a_no_op(monkeypatch):
    monkeypatch.setattr(lc.config, "LANGFUSE_ENABLED", False)
    lc.get_callbacks("s1")
    lc.flush()  # must not raise
