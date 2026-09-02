"""visit_webpage - fetch one URL and return its readable text.

Adapted from the GAIA project's version, which worked nearly as-is. Two
changes: the result goes through the Observation contract, and the bare
`f"Couldn't fetch '{url}': {e}"` return is now an explicit failure. That string
was the exact problem this project designs against - it reads to the model as
content rather than as a failure, and nothing downstream could tell them apart.

Keeps the `query` parameter: long pages get truncated, and centring the excerpt
on a keyword is how you reach a section that sits below the cut.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from markdownify import markdownify

from myassistant.tools.observation import failed, fetched

MAX_LENGTH = 8000  # roughly what a 3B context can absorb without losing the question
WINDOW = 4000  # excerpt size when centring on a query match
LEAD = 500  # characters kept before the match, for context
TIMEOUT_S = 15

# Real browser UA: a default python-requests UA is blocked outright by enough
# sites that it would look like a content problem rather than a client one.
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _clean(html: str) -> str:
    """Strip chrome, then convert to markdown.

    Dropping nav/header/footer/aside before conversion matters more than it
    looks: on a thin page they can be most of the text, which would push a
    near-empty article over the usable-content threshold.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()
    return markdownify(str(soup)).strip()


def _fit(text: str, query: str) -> tuple[str, bool]:
    """Cut the page to something a small model can hold. Returns (text, truncated)."""
    if len(text) <= MAX_LENGTH:
        return text, False

    # With a query, centre the window on the first match so a section below the
    # cut is still reachable without a second fetch.
    if query:
        idx = text.lower().find(query.lower())
        if idx != -1:
            start = max(0, idx - LEAD)
            excerpt = text[start : min(len(text), idx + WINDOW)]
            return f"...[excerpt around {query!r}]...\n{excerpt}\n...[truncated]", True

    return text[:MAX_LENGTH] + "\n...[truncated - pass a query to search deeper]", True


@tool
def visit_webpage(url: str, query: str = "") -> str:
    """Fetch a web page and return its text content.

    Use after web_search when a snippet suggests the page holds the answer.
    Long pages are truncated; if what you need is likely further down (a named
    section, a table, a heading), pass that as `query` to get an excerpt
    centred on the first match instead of just the top of the page.

    Args:
        url: full URL to fetch, e.g. from a web_search result
        query: optional keyword or phrase to centre the excerpt on
    """
    try:
        resp = requests.get(url, timeout=TIMEOUT_S, headers=_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Status codes live in metrics so the model sees 403 vs 404 vs timeout
        # and can decide whether a different URL is worth trying.
        status = getattr(getattr(e, "response", None), "status_code", None)
        return failed(
            f"could not fetch the page: {e}", source=url, kind="page", status=status
        ).render()

    text = _clean(resp.text)
    payload, truncated = _fit(text, query)

    # Truncating before fetched() is deliberate: the excerpt is still far above
    # the usable-content threshold, and block-page markers sit near the top, so
    # detection still works - while `full_chars` records what the page really had.
    return fetched(
        payload,
        source=url,
        kind="page",
        status=resp.status_code,
        full_chars=len(text.strip()),
        truncated=truncated,
    ).render()
