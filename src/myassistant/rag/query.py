"""Search one collection, and say honestly how well it matched.

    search(RESUME_INTERVIEW, "what did I do at Capital One?")
       |
       v
    embed the question -> nearest chunks, each with a relevance score
       |
       v
    top score >= threshold ?
       |                 |
      yes                no
       |                 |
    ok=True          ok=False, but the chunks are still returned
    (HIGH)           (LOW - "my notes don't really cover this")

The score is the whole point. It is a real number produced by the retrieval
that already happened, not a model's opinion of its own answer - which is what
the confidence rule means by evidence-based. Returning the chunks even when
they score badly is deliberate: a weak match is sometimes still the right
answer, and the tier is what stops it being presented as a strong one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from myassistant import config
from myassistant.rag.store import Collection, get
from myassistant.tools.observation import Observation, failed


def _format(hits: list[tuple[Document, float]]) -> str:
    """One block per chunk, each naming the file it came from.

    The filename is included so the answer can say where it got something -
    "your Michelin STAR doc says..." is checkable in a way that unattributed
    prose is not.
    """
    return "\n\n".join(
        f"[{doc.metadata.get('name', 'unknown')} · relevance {score:.2f}]\n{doc.page_content.strip()}"
        for doc, score in hits
    )


def search(
    collection: Collection,
    question: str,
    *,
    k: int | None = None,
    threshold: float | None = None,
    embedding_function: Any | None = None,
) -> Observation:
    """Retrieve from one collection and score the result.

    Never raises: a missing collection, an unreachable embedding model and an
    empty store all come back as explicit failures, because the agent can only
    react to what it can see in the observation text.
    """
    k = k or config.RAG_TOP_K
    threshold = config.RAG_RELEVANCE_THRESHOLD if threshold is None else threshold

    try:
        hits = get(
            collection, embedding_function=embedding_function
        ).similarity_search_with_relevance_scores(question, k=k)
    except Exception as e:  # noqa: BLE001 - degrade loudly, never kill the turn
        return failed(f"could not search {collection}: {e}", source=str(collection), kind="notes")

    if not hits:
        # An empty collection and a genuine no-match are different problems, and
        # "nothing found" alone would read to the model as a settled answer.
        return failed(
            f"nothing in {collection} matched - the collection may be empty, "
            f"or these notes may not cover it",
            source=str(collection),
            kind="notes",
            n_results=0,
        )

    # Not guaranteed to be in [0, 1]: Chroma derives relevance from the
    # collection's distance metric, and an unrelated document can score
    # negative (measured -0.41 against an orthogonal vector; Chroma warns about
    # it). Harmless here - a threshold comparison works on any ordered scale -
    # but it means the number should be read as "higher is better", not as a
    # probability, which is the same reason the tiers are not percentages.
    top = max(score for _, score in hits)
    metrics = {
        "kind": "notes",
        "n_results": len(hits),
        "top_score": round(top, 3),
        "threshold": threshold,
    }

    if top < threshold:
        # Still hand back the chunks. A weak match can be the right answer, and
        # the model should be able to see it and say so - but ok=False means the
        # gate scores this LOW, so it can never be presented as well-grounded.
        return Observation(
            ok=False,
            detail=f"only weak matches in {collection} (best {top:.2f} < {threshold})",
            content=_format(hits),
            source=str(collection),
            metrics=metrics,
        )

    return Observation(
        ok=True,
        detail=f"{len(hits)} matches from {collection}",
        content=_format(hits),
        source=str(collection),
        metrics=metrics,
    )
