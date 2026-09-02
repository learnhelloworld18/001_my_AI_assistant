"""visit_webpage: the silent-failure cases are the ones that matter.

The GAIA version returned f"Couldn't fetch '{url}': {e}" on error - a plain
string the model reads as content. These tests exist to keep that from
coming back.
"""

import pytest
import requests

from myassistant.tools import visit_webpage as vw

ARTICLE = "<p>" + ("Kafka keeps an in-sync replica set per partition. " * 40) + "</p>"


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Client Error")
            err.response = self
            raise err


def _install(monkeypatch, resp=None, error=None):
    def fake_get(url, **kwargs):
        if error:
            raise error
        return resp

    monkeypatch.setattr(vw.requests, "get", fake_get)


def _run(url="https://example.com", query=""):
    return vw.visit(url, query).render()


def test_a_real_page_comes_back_ok(monkeypatch):
    _install(monkeypatch, _Resp(ARTICLE))
    out = _run()
    assert out.startswith("[OK]")
    assert "in-sync replica" in out
    assert "kind=page" in out


def test_a_403_is_an_explicit_failure_with_its_status(monkeypatch):
    """403 vs 404 vs timeout changes whether another URL is worth trying."""
    _install(monkeypatch, _Resp("<p>Forbidden</p>", status=403))
    out = _run()
    assert out.startswith("[TOOL FAILED]")
    assert "status=403" in out


def test_a_timeout_does_not_raise_through_the_tool(monkeypatch):
    _install(monkeypatch, error=requests.Timeout("timed out"))
    out = _run()
    assert out.startswith("[TOOL FAILED]")
    assert "timed out" in out


def test_a_200_consent_wall_is_a_failure_not_content(monkeypatch):
    """The dangerous case: HTTP says fine, the body is a block page."""
    _install(monkeypatch, _Resp("<p>" + "Please enable JavaScript to continue. " * 30 + "</p>"))
    out = _run()
    assert out.startswith("[TOOL FAILED]")


def test_a_200_with_almost_no_text_is_a_failure(monkeypatch):
    """A JS shell returns 200 and nothing readable."""
    _install(monkeypatch, _Resp("<html><body><div id='root'></div></body></html>"))
    out = _run()
    assert out.startswith("[TOOL FAILED]")


def test_chrome_is_stripped_before_the_emptiness_check(monkeypatch):
    """Nav and footer text could push a near-empty article over the threshold."""
    page = (
        "<nav>" + ("Home About Contact Careers Blog " * 40) + "</nav>"
        "<p>Short.</p>"
        "<footer>" + ("Terms Privacy Cookies " * 40) + "</footer>"
    )
    _install(monkeypatch, _Resp(page))
    assert _run().startswith("[TOOL FAILED]")


def test_long_pages_are_truncated_and_say_so(monkeypatch):
    _install(monkeypatch, _Resp("<p>" + ("word " * 5000) + "</p>"))
    out = _run()
    assert out.startswith("[OK]")
    assert "truncated=True" in out


def test_query_centres_the_excerpt_on_a_match_below_the_cut(monkeypatch):
    """Without this, a section past 8000 chars is unreachable without refetching."""
    body = ("filler " * 3000) + " CHECKPOINTING SEMANTICS " + ("filler " * 3000)
    _install(monkeypatch, _Resp(f"<p>{body}</p>"))
    out = _run(query="checkpointing semantics")
    assert "CHECKPOINTING SEMANTICS" in out
    assert "excerpt around" in out


def test_full_chars_records_the_real_page_size(monkeypatch):
    """chars reports what was returned; full_chars what the page actually had."""
    _install(monkeypatch, _Resp("<p>" + ("word " * 5000) + "</p>"))
    out = _run()
    assert "full_chars=" in out


@pytest.mark.live
def test_live_fetch_of_a_real_page():
    """Catches the class of bug a mocked suite cannot - see the ddgs incident."""
    out = _run("https://example.com")
    assert out.startswith(("[OK]", "[TOOL FAILED]"))
