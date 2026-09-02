"""web_search - Tavily search, returning snippets only.

Rewritten rather than reused: the GAIA project's version was DuckDuckGo-based
(flaky DNS, inconsistent snippets). Tavily is LLM-oriented and more reliable.

Snippets are deliberately not treated as strong evidence. A search result is a
claim *about* a page, not the page - so every Observation here is tagged
kind="search", and the confidence gate will not award HIGH on search alone.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from myassistant import config
from myassistant.tools.observation import Observation, failed

# Low on purpose: more results is more context for a 3B model to lose the
# thread in, and the useful next step is visiting one page, not skimming ten.
MAX_RESULTS = 5

_client: TavilySearch | None = None


def _search() -> TavilySearch:
    """Build the Tavily client once, lazily.

    Lazy because TavilySearch reads TAVILY_API_KEY at construction - building it
    at import time would fail for anyone who imports this module without a key.
    """
    global _client
    if _client is None:
        _client = TavilySearch(max_results=MAX_RESULTS)
    return _client


def _format(results: list[dict[str, Any]]) -> str:
    """One numbered block per result, url on its own line so it is copyable."""
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"{i}. {r.get('title', '(no title)')}\n   {r.get('url', '')}\n   {r.get('content', '').strip()}"
        )
    return "\n\n".join(blocks)


@tool
def web_search(query: str) -> str:
    """Search the web and return short snippets from the top results.

    Returns titles, URLs and a couple of sentences each - not full pages. When
    a snippet looks like it holds the answer, call visit_webpage on its URL to
    read the actual page. Use for current events, product news, and finding
    which page to read. Not for official cloud or library documentation, which
    has its own tools.

    Args:
        query: what to search for, in plain words
    """
    # A missing key is a configuration failure, not an empty result set. Saying
    # so plainly is the difference between the model retrying pointlessly and
    # the model reporting that search is unavailable.
    if not config.TAVILY_ENABLED:
        return failed("web search unavailable: TAVILY_API_KEY is not configured").render()

    try:
        raw = _search().invoke({"query": query})
    except Exception as e:  # noqa: BLE001 - any failure must surface as a loud Observation
        return failed(f"web search failed: {e}", metrics={"kind": "search"}).render()

    results = raw.get("results", []) if isinstance(raw, dict) else []

    # Never return "" or "No results found" - both read to the model as a
    # legitimate empty answer rather than something to react to.
    if not results:
        return failed(
            f"web search returned no results for {query!r} - try different wording",
            kind="search",
            n_results=0,
        ).render()

    return Observation(
        ok=True,
        detail=f"{len(results)} search results (snippets only - visit_webpage for full content)",
        content=_format(results),
        metrics={"kind": "search", "n_results": len(results)},
    ).render()
