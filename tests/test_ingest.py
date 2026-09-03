"""ingest.py: the exclusions and the idempotence.

Most of these encode a decision made after looking at a real documents folder,
where a naive walk pulled in a cloned repo, stale backups and certificates.
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from myassistant import config
from myassistant.rag import ingest, store

RESUME = "Led the migration from EMR to Glue, cutting pipeline cost by 40%. " * 30


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real vector store or manifest."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")


@pytest.fixture
def fake():
    return FakeEmbeddings(size=32)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "manifest.db"


@pytest.fixture
def docs(tmp_path):
    """A folder shaped like the real one, including everything to exclude."""
    root = tmp_path / "docs"
    (root / "PREP" / "STAR").mkdir(parents=True)
    (root / "PREP" / "repo" / ".git").mkdir(parents=True)
    (root / "PREP" / "spark.md").write_text(RESUME)
    (root / "PREP" / "STAR" / "star_michelin.md").write_text(RESUME)
    (root / "PREP" / "STAR" / "star_michelin_BACKUP_2026-08-09.md").write_text(RESUME)
    (root / "PREP" / "repo" / "README.md").write_text(RESUME)
    (root / "PREP" / "repo" / "main.py").write_text("print('hi')")
    (root / "AWS Certified Data Engineer.pdf").write_text("cert")
    (root / "CLAUDE.md").write_text(RESUME)
    (root / "Google Services authentication.pdf").write_text(RESUME)
    (root / "resume.pages").write_text("binary-ish")
    (root / ".DS_Store").write_text("junk")
    return root


def _names(paths):
    return sorted(p.name for p in paths)


# --- what gets picked up ---


def test_only_real_notes_are_discovered(docs):
    assert _names(ingest.discover(docs)) == ["spark.md", "star_michelin.md"]


def test_backup_copies_are_excluded(docs):
    """A dated backup beside the live file puts a stale answer in retrieval."""
    assert not any("BACKUP" in p.name for p in ingest.discover(docs))


def test_a_nested_git_repo_is_excluded_including_its_readme(docs):
    """Code belongs to coding_agent's file tools, not the vector store."""
    found = _names(ingest.discover(docs))
    assert "README.md" not in found
    assert "main.py" not in found


def test_pages_files_are_excluded(docs):
    """Apple bundles hold no extractable text - only JPG previews."""
    assert not any(p.suffix == ".pages" for p in ingest.discover(docs))


def test_certificates_and_tool_instructions_are_excluded(docs):
    found = _names(ingest.discover(docs))
    assert not any("Certified" in n for n in found)
    assert "CLAUDE.md" not in found


def test_named_credential_files_are_excluded(docs):
    """A vector store is somewhere a model can quote from - keep secrets out."""
    assert "Google Services authentication.pdf" not in _names(ingest.discover(docs))


def test_dotfiles_are_excluded(docs):
    assert not any(p.name.startswith(".") for p in ingest.discover(docs))


def test_a_single_file_can_be_ingested(docs):
    assert _names(ingest.discover(docs / "PREP" / "spark.md")) == ["spark.md"]


def test_an_excluded_single_file_yields_nothing(docs):
    assert ingest.discover(docs / "CLAUDE.md") == []


def test_discovery_is_ordered(docs):
    """Reproducible runs - the same folder always ingests in the same order."""
    assert ingest.discover(docs) == sorted(ingest.discover(docs))


# --- ingesting ---


def test_ingest_stores_chunks_with_their_source(docs, fake, db):
    result = ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert len(result.added) == 2
    assert result.chunks > 0
    col = store.get(store.Collection.TECH_NOTES, embedding_function=fake)
    sources = {m["source"] for m in col._collection.get()["metadatas"]}
    assert len(sources) == 2


def test_re_running_skips_unchanged_files(docs, fake, db):
    """The second run must be nearly free - that is the point of the manifest."""
    ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    second = ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert second.added == []
    assert len(second.skipped) == 2


def _chunks_per_source(fake):
    """How many chunks each file currently has in the store."""
    got = store.get(store.Collection.TECH_NOTES, embedding_function=fake)._collection.get()
    counts: dict[str, int] = {}
    for meta in got["metadatas"]:
        counts[meta["name"]] = counts.get(meta["name"], 0) + 1
    return counts


def test_editing_a_file_replaces_its_chunks_rather_than_adding(docs, fake, db):
    """The whole reason ingestion is not append-only."""
    ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    before = _chunks_per_source(fake)
    assert before["spark.md"] > 1  # long file, several chunks

    # Long enough to clear MIN_USEFUL_CHARS, short enough to be a single chunk.
    (docs / "PREP" / "spark.md").write_text(
        "Rewrote the whole ingestion pipeline in Snowflake instead of Spark, "
        "which removed the cluster entirely."
    )
    ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)

    after = _chunks_per_source(fake)
    assert after["spark.md"] == 1  # replaced, not added to
    assert after["star_michelin.md"] == before["star_michelin.md"]  # untouched file unaffected


def test_the_old_wording_does_not_survive_an_edit(docs, fake, db):
    """Stale and current answers competing in retrieval is the bug being prevented."""
    single = docs / "PREP" / "spark.md"
    ingest.ingest(single, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    single.write_text("Rewrote the pipeline in Snowflake. " * 30)
    ingest.ingest(single, store.Collection.TECH_NOTES, embedding_function=fake, db=db)

    texts = " ".join(
        store.get(store.Collection.TECH_NOTES, embedding_function=fake)._collection.get()[
            "documents"
        ]
    )
    assert "Snowflake" in texts
    assert "EMR to Glue" not in texts


def test_force_reingests_unchanged_files(docs, fake, db):
    """Needed after changing CHUNK_SIZE: same files, different chunking."""
    ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    forced = ingest.ingest(
        docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db, force=True
    )
    assert len(forced.added) == 2


def test_collections_are_tracked_separately(docs, fake, db):
    """The same file ingested into two collections is two records, not one."""
    ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    second = ingest.ingest(docs, store.Collection.RESUME_INTERVIEW, embedding_function=fake, db=db)
    assert len(second.added) == 2  # not skipped - a different collection


def test_a_very_short_document_is_reported_not_stored(tmp_path, fake, db):
    """MIN_USEFUL_CHARS also catches near-empty parses, not just blank ones."""
    root = tmp_path / "short"
    root.mkdir()
    (root / "stub.md").write_text("too short")
    result = ingest.ingest(root, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert result.added == []
    assert "no extractable text" in result.failed[0][1]


def test_an_empty_document_is_reported_not_stored(tmp_path, fake, db):
    """A scanned PDF parses fine and yields nothing. That must be said out loud."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "scanned.md").write_text("   ")  # a scanned PDF parses to whitespace
    result = ingest.ingest(root, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert result.added == []
    assert "no extractable text" in result.failed[0][1]


def test_one_unreadable_file_does_not_stop_the_run(docs, fake, db, monkeypatch):
    """Ingesting 77 files must not be all-or-nothing."""
    real_load = ingest.load

    def explode(path):
        if path.name == "spark.md":
            raise OSError("disk gremlin")
        return real_load(path)

    monkeypatch.setattr(ingest, "load", explode)
    result = ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert len(result.added) == 1
    assert len(result.failed) == 1


def test_a_failure_is_named_in_the_summary(docs, fake, db, monkeypatch):
    monkeypatch.setattr(ingest, "load", lambda p: (_ for _ in ()).throw(OSError("nope")))
    result = ingest.ingest(docs, store.Collection.TECH_NOTES, embedding_function=fake, db=db)
    assert "spark.md" in result.summary()
    assert "nope" in result.summary()


def test_chunks_carry_the_source_path_for_deletion_and_citation(tmp_path):
    pieces = ingest.chunk(RESUME, tmp_path / "a.md")
    assert all(p.metadata["source"] == str(tmp_path / "a.md") for p in pieces)
    assert all(p.metadata["name"] == "a.md" for p in pieces)


def test_chunks_overlap_so_a_split_fact_survives(tmp_path):
    """A sentence straddling a boundary must appear whole in one of the chunks."""
    pieces = ingest.chunk("word " * 1000, tmp_path / "x.md")
    assert len(pieces) > 1
    assert ingest.CHUNK_OVERLAP > 0
