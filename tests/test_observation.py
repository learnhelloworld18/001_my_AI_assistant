"""observation.py: the point is that silent failures become loud ones.

These target the GAIA-class bug directly - a call that "succeeds" and returns
something superficially fine, which the model then has no way to notice.
"""

from dataclasses import FrozenInstanceError

import pytest

from myassistant.tools.observation import (
    MIN_USEFUL_CHARS,
    Observation,
    failed,
    fetched,
    looks_empty,
)

REAL_PAGE = "Kubernetes schedules containers across a cluster. " * 40


def test_real_content_passes():
    assert looks_empty(REAL_PAGE) is None


def test_short_body_is_not_usable():
    """A 200 response with 80 chars is a failure wearing a success's clothes."""
    why = looks_empty("Loading...")
    assert why is not None
    assert "chars extracted" in why


def test_whitespace_does_not_count_as_content():
    assert looks_empty(" \n\t" * 500) is not None


def test_block_pages_are_rejected_despite_being_long_enough():
    """The dangerous case: long enough to pass a length check, still not content."""
    for body in (
        "Please enable JavaScript to continue. " * 20,
        "Access Denied. You do not have permission. " * 20,
        "Verify you are human before continuing. " * 20,
    ):
        assert len(body) > MIN_USEFUL_CHARS
        assert looks_empty(body) is not None


def test_marker_only_checked_near_the_start():
    """A page legitimately about captchas shouldn't be discarded for saying so."""
    body = REAL_PAGE + " " * 3000 + "captcha"
    assert looks_empty(body) is None


def test_fetched_downgrades_a_block_page_to_failure():
    obs = fetched("Please enable JavaScript. " * 20, source="https://example.com", status=200)
    assert not obs.ok
    assert obs.content == ""  # nothing plausible-looking leaks through
    assert obs.metrics["status"] == 200


def test_fetched_keeps_real_content():
    obs = fetched(REAL_PAGE, source="https://example.com", status=200)
    assert obs.ok
    assert obs.content == REAL_PAGE
    assert obs.metrics["chars"] == len(REAL_PAGE.strip())


def test_failure_renders_an_unmissable_marker():
    """The model's only signal is this text, so failure must be lexically obvious."""
    rendered = failed("HTTP 403", source="https://example.com", status=403).render()
    assert rendered.startswith("[TOOL FAILED]")
    assert "status=403" in rendered
    assert "https://example.com" in rendered


def test_success_renders_its_numbers():
    """Metrics are what let the next reasoning step notice a thin result."""
    rendered = fetched(REAL_PAGE, source="https://example.com", status=200).render()
    assert rendered.startswith("[OK]")
    assert f"chars={len(REAL_PAGE.strip())}" in rendered
    assert REAL_PAGE in rendered


def test_observation_is_immutable():
    """Evidence a confidence tier is derived from must not be rewritten downstream."""
    obs = Observation(ok=True, detail="fine")
    with pytest.raises(FrozenInstanceError):
        obs.ok = False  # type: ignore[misc]
