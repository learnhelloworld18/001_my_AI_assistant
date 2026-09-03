"""Chroma collections - the one place that knows where vectors live.

Chroma is embedded, not a server: it is a library that reads and writes files
under ASSISTANT_HOME. Nothing to start, nothing that can be "down". That was
the point of choosing it - Langfuse is the single accepted exception to this
project's no-server rule, and it earns that by being optional.

Collections are separate rather than one store with a metadata filter, because
docs_agent gets one tool per collection: the model's choice of tool is what
says which body of knowledge to search, and that is a decision a small model
makes far more reliably than a filter argument.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from myassistant import config


class Collection(StrEnum):
    """The three bodies of knowledge, kept apart on purpose."""

    TECH_NOTES = "tech_notes"  # general technical reference - Spark, Kafka, docs
    RESUME_INTERVIEW = "resume_interview"  # your CV, STAR answers, interview prep
    CONVERSATION_MEMORY = "conversation_memory"  # session summaries (step 6)


def embeddings() -> OllamaEmbeddings:
    """nomic-embed-text via Ollama. An embedding model, not a chat model.

    Built fresh each call rather than cached: it holds no meaningful state, and
    a module-level instance would open a client at import time.
    """
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_HOST)


def get(collection: Collection, embedding_function: Any | None = None) -> Chroma:
    """Open (or create) one collection. The directory appears on first write.

    `embedding_function` is injectable so tests can use a fake and stay off
    Ollama - embedding two documents is fast, but a test suite that needs a
    model running is a live test wearing a component test's clothes.
    """
    return Chroma(
        collection_name=str(collection),
        embedding_function=embedding_function or embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def drop_source(collection: Collection, source: str, **kwargs: Any) -> None:
    """Remove every chunk that came from one file.

    This is what makes re-ingestion idempotent rather than append-only. Without
    it, editing a resume bullet and re-ingesting would leave the old wording in
    the store competing with the new one - and retrieval would happily return
    both, with no way to tell which is current.
    """
    get(collection, **kwargs).delete(where={"source": source})
