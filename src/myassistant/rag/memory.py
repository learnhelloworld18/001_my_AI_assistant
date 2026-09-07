"""Cross-session memory - what the assistant remembers after you close it.

    /exit                        /remember <text>        new session
      |                            |                       |
      v                            v                       v
    summarise the session        store verbatim          recall relevant
    with a small model             |                     summaries
      |                            |                       |
      +--------> conversation_memory (Chroma) <------------+

Summaries, not transcripts. That is the whole design decision here: a raw
conversation is mostly plumbing - tool output, restated questions, the
supervisor's throat-clearing - and embedding it fills retrieval with noise
that competes with the parts worth recalling. One distilled paragraph per
session retrieves far better than fifty turns of it.

`/remember` skips the summarising step because you already did it. Text you
chose to write down is more signal than anything a 3B would distil from a
transcript, so it is stored exactly as given.

Never touches tech_notes or resume_interview. Those have their own deliberate
ingestion path through /ingest, and mixing a session's chatter into them would
quietly degrade the collections that answer your interview questions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from myassistant import config
from myassistant.rag.store import Collection, get

log = logging.getLogger("myassistant")

# Below this a session had nothing worth keeping - a single "hello", or one
# question and a failed answer. Storing those would dilute retrieval with
# entries that can only ever be noise.
MIN_TURNS = 2

# Recall is deliberately narrow. Past sessions are context, not the answer, and
# a 3B has little room to spare for them.
RECALL_K = 3

SUMMARY_PROMPT = """Summarise this conversation in 2-4 sentences, for your own \
future reference.

Record what the user asked about, what was decided or concluded, and anything \
they said about themselves, their work or their preferences. Write plainly, in \
the third person, as notes rather than prose.

Leave out pleasantries, failed attempts and anything you could not answer - a \
note saying you did not know something is worse than no note."""


def _model() -> ChatOllama:
    """The small model. Summarising happens on exit, where latency is cheapest."""
    return ChatOllama(
        model=config.GENERAL_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SUPERVISOR_KEEP_ALIVE,
    )


def _transcript(history: list[BaseMessage]) -> str:
    """The conversation as plain text, for the summariser to read."""
    lines = []
    for message in history:
        who = "User" if isinstance(message, HumanMessage) else "Assistant"
        text = str(message.content).strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines)


def store(text: str, session_id: str, *, kind: str, embedding_function: Any | None = None) -> bool:
    """Put one note in conversation_memory. Returns whether anything was stored.

    `kind` distinguishes a summary from a deliberate note, so a later change of
    mind about summaries does not have to throw away what you wrote by hand.
    """
    text = text.strip()
    if not text:
        return False
    try:
        get(Collection.CONVERSATION_MEMORY, embedding_function=embedding_function).add_documents(
            [
                Document(
                    page_content=text,
                    metadata={
                        "kind": kind,
                        "session": session_id,
                        "at": datetime.now(UTC).isoformat(),
                        # Every other collection carries `source`; keeping the
                        # key uniform means one metadata filter works everywhere.
                        "source": f"memory:{session_id}",
                        "name": f"{kind} from {datetime.now(UTC):%Y-%m-%d}",
                    },
                )
            ]
        )
    except Exception:
        log.exception("could not store memory")
        return False
    return True


def remember(text: str, session_id: str, **kwargs: Any) -> bool:
    """/remember <text> - store it exactly as written, no summarising.

    Verbatim on purpose: you already distilled it, and a 3B paraphrasing a
    sentence you chose carefully can only lose something.
    """
    return store(text, session_id, kind="note", **kwargs)


def summarise_session(
    history: list[BaseMessage],
    session_id: str,
    *,
    model: Any | None = None,
    **kwargs: Any,
) -> str | None:
    """Summarise a finished session and store it. Returns the summary, or None.

    Called on the way out, where a few seconds cost nothing - the user has
    already stopped waiting.
    """
    if len(history) < MIN_TURNS:
        return None

    try:
        reply = (model or _model()).invoke(
            [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=_transcript(history))]
        )
        summary = str(reply.content).strip()
    except Exception:
        log.exception("could not summarise the session")
        return None

    return summary if store(summary, session_id, kind="summary", **kwargs) else None


def recall(query: str, *, k: int | None = None, embedding_function: Any | None = None) -> list[str]:
    """Relevant notes from past sessions. Empty on a first run, or on failure.

    Deliberately returns plain strings rather than an Observation: this is
    context handed to a prompt, not evidence a confidence tier is derived from.
    What the assistant remembers is not a source it checked.

    Uses MEMORY_RECALL_THRESHOLD, not the document one - see config for the
    measurements. The document threshold recalled nothing at all.
    """
    try:
        hits = get(
            Collection.CONVERSATION_MEMORY, embedding_function=embedding_function
        ).similarity_search_with_relevance_scores(query, k=k or RECALL_K)
    except Exception:
        log.exception("could not recall memories")
        return []
    return [doc.page_content for doc, score in hits if score >= config.MEMORY_RECALL_THRESHOLD]
