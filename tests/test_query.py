"""query.py: the relevance score is the evidence, and it must be honest."""

import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from myassistant import config
from myassistant.rag import query, store
from myassistant.state import ConfidenceTier, tier_from_observations


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")


class _KeywordEmbeddings(Embeddings):
    """One dimension per keyword, unit-normalised.

    langchain_core's FakeEmbeddings returns random vectors, so "similar" text
    is not similar and relevance scores come back at -59 - Chroma warns they
    must be in [0, 1]. Useless for testing a threshold. Here word overlap *is*
    cosine similarity, so an exact-topic match scores near 1 and an unrelated
    document near 0, which is the shape real embeddings produce.
    """

    VOCAB = ("broadcast", "join", "spark", "executor", "kafka", "replica", "partition")

    def _vector(self, text: str) -> list[float]:
        raw = [1.0 if word in text.lower() else 0.0 for word in self.VOCAB]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake():
    return _KeywordEmbeddings()


@pytest.fixture
def notes(fake):
    col = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    col.add_documents(
        [
            Document(
                page_content="Broadcast joins ship the small side to every executor.",
                metadata={"source": "/p/spark.md", "name": "spark.md"},
            ),
            Document(
                page_content="Kafka keeps an in-sync replica set per partition.",
                metadata={"source": "/p/kafka.md", "name": "kafka.md"},
            ),
        ]
    )
    return col


def _search(fake, **kw):
    return query.search(
        store.Collection.TECH_NOTES, "how do broadcast joins work?", embedding_function=fake, **kw
    )


def test_an_empty_collection_is_an_explicit_failure(fake):
    """ "Nothing found" alone reads to a model as a settled answer. It is not."""
    obs = _search(fake)
    assert not obs.ok
    assert "may be empty" in obs.detail
    assert obs.render().startswith("[TOOL FAILED]")


def test_a_good_match_is_ok_and_carries_its_score(notes, fake):
    obs = _search(fake, threshold=-1.0)  # accept anything, to test the shape
    assert obs.ok
    assert obs.metrics["top_score"] is not None
    assert obs.metrics["kind"] == "notes"
    assert obs.metrics["n_results"] > 0


def test_a_weak_match_is_not_ok_but_is_still_returned(notes, fake):
    """A weak match can still be the right answer - the tier stops it being
    presented as a strong one. Hiding it would be worse than labelling it."""
    obs = _search(fake, threshold=1.1)  # nothing can clear this
    assert not obs.ok
    assert obs.content  # the chunks are still there for the model to judge
    assert "weak matches" in obs.detail


def test_results_name_the_file_they_came_from(notes, fake):
    """An answer that cites your Michelin STAR doc is checkable; prose is not."""
    obs = _search(fake, threshold=-1.0)
    assert "spark.md" in obs.content or "kafka.md" in obs.content
    assert "relevance" in obs.content


def test_a_broken_embedding_model_degrades_instead_of_raising(notes, monkeypatch):
    """Ollama being down must not kill the turn."""

    class _Broken:
        def embed_query(self, text):
            raise ConnectionError("ollama is not running")

        def embed_documents(self, texts):
            raise ConnectionError("ollama is not running")

    obs = query.search(store.Collection.TECH_NOTES, "anything", embedding_function=_Broken())
    assert not obs.ok
    assert "could not search" in obs.detail


def test_k_limits_how_much_context_is_returned(notes, fake):
    """A 3B window filled with marginal chunks answers worse than one good one."""
    assert _search(fake, k=1, threshold=-1.0).metrics["n_results"] == 1


# --- how the tier reads these ---


def test_a_good_match_earns_high(notes, fake):
    assert tier_from_observations([_search(fake, threshold=-1.0)]) is ConfidenceTier.HIGH


def test_a_weak_match_stays_low(notes, fake):
    """The spec: below threshold the answer must say the notes don't cover it."""
    assert tier_from_observations([_search(fake, threshold=1.1)]) is ConfidenceTier.LOW


def test_an_empty_collection_stays_low(fake):
    assert tier_from_observations([_search(fake)]) is ConfidenceTier.LOW


def test_the_default_threshold_comes_from_config(notes, fake):
    """One place to tune it once real numbers exist."""
    assert _search(fake).metrics["threshold"] == config.RAG_RELEVANCE_THRESHOLD


# --- cross-role search ---


def test_cross_role_search_says_when_it_is_unconfigured(monkeypatch, fake):
    """Unconfigured and "nothing matched" are different problems, and only one
    of them is fixed by editing .env."""
    monkeypatch.setattr(config, "CAREER_ROLES", ())
    obs = query.search_across_roles("walk me through my career", embedding_function=fake)
    assert not obs.ok
    assert "CAREER_ROLES" in obs.detail


def test_cross_role_search_covers_every_role(monkeypatch, fake):
    """The whole point: no role can be silently missing from the answer."""
    monkeypatch.setattr(config, "CAREER_ROLES", (("Alpha", ("alpha",)), ("Beta", ("beta",))))
    col = store.get(store.Collection.RESUME_INTERVIEW, embedding_function=fake)
    col.add_documents(
        [
            Document(
                page_content="Broadcast joins at Alpha.",
                metadata={"source": "/a.md", "name": "a.md", "role": "Alpha"},
            ),
            Document(
                page_content="Kafka partitions at Beta.",
                metadata={"source": "/b.md", "name": "b.md", "role": "Beta"},
            ),
        ]
    )
    obs = query.search_across_roles("broadcast join", embedding_function=fake, threshold=-1.0)
    assert obs.ok
    assert "## Alpha" in obs.content
    assert "## Beta" in obs.content
    assert obs.metrics["roles_covered"] == 2


def test_roles_appear_in_configured_priority_order(monkeypatch, fake):
    """Most important first, so a truncated answer loses the least."""
    monkeypatch.setattr(config, "CAREER_ROLES", (("Alpha", ("alpha",)), ("Beta", ("beta",))))
    col = store.get(store.Collection.RESUME_INTERVIEW, embedding_function=fake)
    col.add_documents(
        [
            Document(
                page_content="spark at Alpha",
                metadata={"source": "/a.md", "name": "a.md", "role": "Alpha"},
            ),
            Document(
                page_content="spark at Beta",
                metadata={"source": "/b.md", "name": "b.md", "role": "Beta"},
            ),
        ]
    )
    content = query.search_across_roles("spark", embedding_function=fake, threshold=-1.0).content
    assert content.index("## Alpha") < content.index("## Beta")


def test_an_empty_role_is_named_rather_than_dropped(monkeypatch, fake):
    """An empty section is information; its absence looks like an omission."""
    monkeypatch.setattr(config, "CAREER_ROLES", (("Alpha", ("alpha",)), ("Gamma", ("gamma",))))
    col = store.get(store.Collection.RESUME_INTERVIEW, embedding_function=fake)
    col.add_documents(
        [
            Document(
                page_content="spark at Alpha",
                metadata={"source": "/a.md", "name": "a.md", "role": "Alpha"},
            ),
        ]
    )
    content = query.search_across_roles("spark", embedding_function=fake, threshold=-1.0).content
    assert "## Gamma" in content
    assert "nothing on file" in content


def test_roles_parse_from_the_environment_format():
    assert config._parse_roles("Company A:acme|acmecorp, Company B") == (
        ("Company A", ("acme", "acmecorp")),
        ("Company B", ("companyb",)),
    )
    assert config._parse_roles("") == ()
