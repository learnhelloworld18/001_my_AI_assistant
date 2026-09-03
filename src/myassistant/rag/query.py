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


def search_across_roles(
    question: str,
    *,
    per_role_k: int | None = None,
    threshold: float | None = None,
    embedding_function: Any | None = None,
) -> Observation:
    """Retrieve for every career role, so a cross-role answer cannot omit one.

    Plain top-k ranks by similarity alone, so the best-matching document's
    near-identical chunks crowd out every other source: "walk me through my
    career" came back with four chunks from three files and no mention of two
    employers - while still scoring HIGH, because the tier certifies retrieval
    quality, not coverage.

    One filtered search per role fixes that by construction. Roles are returned
    in config order, most important first, so the answer leads with the most
    recent work and a truncated answer loses the least.

    A role with nothing to say is reported rather than skipped silently: an
    empty section is information, and its absence would look like an omission.
    """
    if not config.CAREER_ROLES:
        # Saying so beats returning nothing: the feature is unconfigured, which
        # is a different problem from "no documents match", and only one of the
        # two is fixed by editing .env.
        return failed(
            "no career roles are configured - set CAREER_ROLES in .env to search "
            "across roles (see .env.example)",
            source=str(Collection.RESUME_INTERVIEW),
            kind="notes",
            n_results=0,
        )

    per_role_k = per_role_k or config.RAG_PER_ROLE_K
    threshold = config.RAG_RELEVANCE_THRESHOLD if threshold is None else threshold
    store = get(Collection.RESUME_INTERVIEW, embedding_function=embedding_function)

    sections: list[str] = []
    scores: list[float] = []
    covered: list[str] = []
    for label, _ in config.CAREER_ROLES:
        try:
            hits = store.similarity_search_with_relevance_scores(
                question, k=per_role_k, filter={"role": label}
            )
        except Exception as e:  # noqa: BLE001 - one bad role must not lose the rest
            sections.append(f"## {label}\n(could not search: {e})")
            continue
        if not hits:
            sections.append(f"## {label}\n(nothing on file for this role)")
            continue
        covered.append(label)
        scores.append(max(score for _, score in hits))
        sections.append(f"## {label}\n{_format(hits)}")

    if not covered:
        return failed(
            "nothing on file for any role - has the resume collection been ingested?",
            source=str(Collection.RESUME_INTERVIEW),
            kind="notes",
            n_results=0,
        )

    top = max(scores)
    return Observation(
        # Coverage is the point here, so ok reflects whether the best role
        # cleared the bar - a thin role is visible in the text either way.
        ok=top >= threshold,
        detail=f"experience across {len(covered)} roles: {', '.join(covered)}",
        content="\n\n".join(sections),
        source=str(Collection.RESUME_INTERVIEW),
        metrics={
            "kind": "notes",
            "roles_covered": len(covered),
            "top_score": round(top, 3),
            "threshold": threshold,
        },
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
