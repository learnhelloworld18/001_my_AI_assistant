"""manifest.py: the record that makes re-ingestion cheap and safe to repeat."""

import time

import pytest

from myassistant.rag import manifest


@pytest.fixture
def db(tmp_path):
    return tmp_path / "manifest.db"


@pytest.fixture
def doc(tmp_path):
    path = tmp_path / "resume.md"
    path.write_text("Led the migration from EMR to Glue.")
    return path


def test_an_unrecorded_file_is_not_unchanged(doc, db):
    assert not manifest.is_unchanged(doc, "resume_interview", db)


def test_a_recorded_file_is_unchanged(doc, db):
    manifest.record(doc, "resume_interview", db)
    assert manifest.is_unchanged(doc, "resume_interview", db)


def test_editing_a_file_makes_it_changed(doc, db):
    manifest.record(doc, "resume_interview", db)
    doc.write_text("Led the migration from EMR to Glue, cutting cost 40%.")
    assert not manifest.is_unchanged(doc, "resume_interview", db)


def test_touching_a_file_does_not_make_it_changed(doc, db):
    """Hash the contents, not the mtime.

    A copy, a restore, or a sync tool all move the timestamp without changing a
    word - and each would otherwise force a full re-embed of the document.
    """
    manifest.record(doc, "resume_interview", db)
    time.sleep(0.01)
    doc.touch()
    assert manifest.is_unchanged(doc, "resume_interview", db)


def test_collections_are_tracked_separately(doc, db):
    """The same file can legitimately live in two collections."""
    manifest.record(doc, "resume_interview", db)
    assert manifest.is_unchanged(doc, "resume_interview", db)
    assert not manifest.is_unchanged(doc, "tech_notes", db)


def test_re_recording_updates_in_place(doc, db):
    """A changed file keeps one row - the manifest must not grow per edit."""
    manifest.record(doc, "resume_interview", db)
    doc.write_text("different content entirely")
    manifest.record(doc, "resume_interview", db)
    assert manifest.sources("resume_interview", db) == [str(doc)]
    assert manifest.is_unchanged(doc, "resume_interview", db)


def test_forget_makes_a_file_new_again(doc, db):
    manifest.record(doc, "resume_interview", db)
    manifest.forget(doc, "resume_interview", db)
    assert not manifest.is_unchanged(doc, "resume_interview", db)
    assert manifest.sources("resume_interview", db) == []


def test_sources_lists_only_that_collection(tmp_path, db):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("a")
    b.write_text("b")
    manifest.record(a, "resume_interview", db)
    manifest.record(b, "tech_notes", db)
    assert manifest.sources("resume_interview", db) == [str(a)]
    assert manifest.sources("tech_notes", db) == [str(b)]


def test_the_first_run_creates_the_table(db):
    """No migration step, no setup command - the file appears on first use."""
    assert not db.exists()
    assert manifest.sources("tech_notes", db) == []
    assert db.exists()


def test_identical_content_hashes_the_same(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("same words")
    b.write_text("same words")
    assert manifest.file_hash(a) == manifest.file_hash(b)
