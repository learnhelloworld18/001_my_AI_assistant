"""memory.py: what survives closing the REPL, and what deliberately doesn't."""

import math

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from myassistant import config
from myassistant.rag import memory, store


class _KeywordEmbeddings(Embeddings):
    """Word overlap as cosine similarity, so relevance is meaningful offline.

    Without this the whole file quietly embeds through real Ollama - a live
    test wearing a component test's clothes, which is exactly what the
    opt-in-only rule forbids.
    """

    VOCAB = ("spark", "kafka", "interview", "engineering", "sourdough", "bread", "prepare")

    def _vector(self, text: str) -> list[float]:
        raw = [1.0 if w in text.lower() else 0.0 for w in self.VOCAB]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No real Chroma directory, and no real embedding model."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(store, "embeddings", _KeywordEmbeddings)


class _Summariser(BaseChatModel):
    reply: str = "User asked about Spark shuffles. Works at a data engineering role."
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "summariser"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


def _history(pairs=2):
    out = []
    for i in range(pairs):
        out += [HumanMessage(content=f"question {i} about spark"), AIMessage(content=f"answer {i}")]
    return out


def _stored():
    return store.get(store.Collection.CONVERSATION_MEMORY)._collection.get()


# --- /remember ---


def test_a_note_is_stored_verbatim():
    """You already distilled it - a 3B paraphrasing it can only lose something."""
    text = "Interviewers keep asking about idempotent pipeline design."
    assert memory.remember(text, "s1")
    assert _stored()["documents"] == [text]


def test_an_empty_note_is_not_stored():
    assert not memory.remember("   ", "s1")
    assert _stored()["documents"] == []


def test_a_note_is_tagged_as_a_note_not_a_summary():
    """So a later change of mind about summaries cannot discard what you wrote."""
    memory.remember("x" * 40, "s1")
    assert _stored()["metadatas"][0]["kind"] == "note"


# --- session summaries ---


def test_a_session_is_summarised_and_stored():
    model = _Summariser()
    summary = memory.summarise_session(_history(), "s1", model=model)
    assert summary
    assert _stored()["metadatas"][0]["kind"] == "summary"


def test_the_summariser_sees_the_conversation_not_the_raw_objects():
    model = _Summariser()
    memory.summarise_session(_history(), "s1", model=model)
    sent = str(model.seen[0][-1].content)
    assert "User: question 0 about spark" in sent
    assert "Assistant: answer 0" in sent


def test_a_trivial_session_is_not_stored():
    """One exchange is noise - storing it would dilute every later recall."""
    assert memory.summarise_session([HumanMessage(content="hi")], "s1") is None
    assert _stored()["documents"] == []


def test_a_failed_summary_does_not_raise():
    """Quitting must not be blocked by a model that is down."""

    class _Dead(_Summariser):
        def _generate(self, *a, **kw):
            raise ConnectionError("ollama is not running")

    assert memory.summarise_session(_history(), "s1", model=_Dead()) is None


# --- recall ---


def test_recall_finds_a_relevant_note():
    memory.remember("The user is preparing for senior data engineering interviews.", "s1")
    assert memory.recall("what interview engineering topics am I preparing?")


def test_recall_ignores_irrelevant_notes():
    """Past sessions are context, not an answer - a weak match is worse than none."""
    memory.remember("The user is preparing for data engineering interviews.", "s1")
    assert memory.recall("how do I bake sourdough bread?") == []


def test_recall_on_a_first_run_is_empty_not_an_error():
    assert memory.recall("anything") == []


def test_recall_returns_plain_strings_not_observations():
    """This is context for a prompt, not evidence a confidence tier reads. What
    the assistant remembers is not a source it checked."""
    memory.remember("The user works with spark and kafka daily.", "s1")
    got = memory.recall("spark kafka")
    assert got and all(isinstance(g, str) for g in got)


# --- collection boundary ---


def test_memory_never_touches_the_other_collections():
    """Session chatter in resume_interview would quietly degrade the collection
    that answers interview questions."""
    memory.remember("something from a session " * 3, "s1")
    for other in (store.Collection.TECH_NOTES, store.Collection.RESUME_INTERVIEW):
        assert store.get(other)._collection.count() == 0
