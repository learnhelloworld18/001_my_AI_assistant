"""store.py: collections, and the delete-before-add that keeps re-ingest honest."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from myassistant import config
from myassistant.rag import store


@pytest.fixture(autouse=True)
def _local_chroma(tmp_path, monkeypatch):
    """Never write to the real ~/.myassistant/chroma from a test."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")


@pytest.fixture
def fake():
    """A deterministic embedding, so these stay component tests, not live ones."""
    return FakeEmbeddings(size=32)


def test_collections_are_distinct_names():
    """One tool per collection, so the names are part of the interface."""
    assert store.Collection.TECH_NOTES == "tech_notes"
    assert store.Collection.RESUME_INTERVIEW == "resume_interview"
    assert store.Collection.CONVERSATION_MEMORY == "conversation_memory"
    assert len(set(store.Collection)) == 3


def test_a_collection_can_be_written_and_read(fake):
    col = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    col.add_documents(
        [Document(page_content="Spark broadcasts small tables.", metadata={"source": "a.md"})]
    )
    assert col._collection.count() == 1


def test_collections_do_not_leak_into_each_other(fake):
    """A resume bullet must never surface as a technical note."""
    notes = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    resume = store.get(store.Collection.RESUME_INTERVIEW, embedding_function=fake)
    notes.add_documents([Document(page_content="Kafka ISR", metadata={"source": "k.md"})])
    assert notes._collection.count() == 1
    assert resume._collection.count() == 0


def test_drop_source_removes_only_that_file(fake):
    """The mechanism behind idempotent re-ingest."""
    col = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    col.add_documents(
        [
            Document(page_content="chunk one", metadata={"source": "a.md"}),
            Document(page_content="chunk two", metadata={"source": "a.md"}),
            Document(page_content="other file", metadata={"source": "b.md"}),
        ]
    )
    store.drop_source(store.Collection.TECH_NOTES, "a.md", embedding_function=fake)
    remaining = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    assert remaining._collection.count() == 1
    assert remaining._collection.get()["metadatas"][0]["source"] == "b.md"


def test_dropping_an_unknown_source_is_not_an_error(fake):
    """First ingest of a file calls this before anything exists."""
    store.drop_source(store.Collection.TECH_NOTES, "never-seen.md", embedding_function=fake)


def test_the_store_lives_under_assistant_home(fake):
    """Not the launch directory - the knowledge base must not fragment."""
    store.get(store.Collection.TECH_NOTES, embedding_function=fake).add_documents(
        [Document(page_content="x", metadata={"source": "x.md"})]
    )
    assert config.CHROMA_DIR.exists()
