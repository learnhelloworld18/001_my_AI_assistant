"""web_search: every path must produce an Observation the gate can read."""

import pytest

from myassistant.tools import web_search as ws


@pytest.fixture(autouse=True)
def _reset_client():
    """The Tavily client is cached at module level."""
    ws._client = None
    yield
    ws._client = None


class _FakeTavily:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if self._error:
            raise self._error
        return self._payload


def _install(monkeypatch, payload=None, error=None, enabled=True):
    fake = _FakeTavily(payload, error)
    monkeypatch.setattr(ws.config, "TAVILY_ENABLED", enabled)
    monkeypatch.setattr(ws, "_search", lambda: fake)
    return fake


def _run(query="spark shuffle"):
    return ws.search(query).render()


RESULTS = {
    "results": [
        {
            "title": "Spark Shuffle",
            "url": "https://a.example",
            "content": "The shuffle writes map output...",
        },
        {
            "title": "Tuning",
            "url": "https://b.example",
            "content": "Set spark.sql.shuffle.partitions...",
        },
    ]
}


def test_missing_key_is_a_configuration_failure_not_an_empty_result(monkeypatch):
    """The model must be able to tell 'search is off' from 'nothing found'."""
    _install(monkeypatch, enabled=False)
    out = _run()
    assert out.startswith("[TOOL FAILED]")
    assert "TAVILY_API_KEY" in out


def test_zero_results_is_a_failure_not_an_empty_string(monkeypatch):
    """ "No results found" reads to a model as a legitimate answer. It is not."""
    _install(monkeypatch, payload={"results": []})
    out = _run()
    assert out.startswith("[TOOL FAILED]")
    assert "no results" in out.lower()


def test_an_exception_becomes_a_loud_observation(monkeypatch):
    _install(monkeypatch, error=RuntimeError("connection reset"))
    out = _run()
    assert out.startswith("[TOOL FAILED]")
    assert "connection reset" in out


def test_results_are_returned_with_urls(monkeypatch):
    """The URLs are the point - they are what visit_webpage acts on next."""
    _install(monkeypatch, payload=RESULTS)
    out = _run()
    assert out.startswith("[OK]")
    assert "https://a.example" in out
    assert "https://b.example" in out


def test_results_are_tagged_as_snippets_not_pages(monkeypatch):
    """kind=search is what stops a search-only turn being scored HIGH."""
    _install(monkeypatch, payload=RESULTS)
    out = _run()
    assert "kind=search" in out
    assert "n_results=2" in out


def test_the_query_is_passed_through(monkeypatch):
    fake = _install(monkeypatch, payload=RESULTS)
    _run("kafka exactly once")
    assert fake.calls == [{"query": "kafka exactly once"}]


def test_a_malformed_response_does_not_raise(monkeypatch):
    """Tavily returning something unexpected must degrade, not crash the turn."""
    _install(monkeypatch, payload="not a dict")
    assert _run().startswith("[TOOL FAILED]")
