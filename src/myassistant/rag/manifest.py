"""What has been ingested, and from which version of which file.

A tiny SQLite table: (source, collection) -> content hash. It answers one
question - "has this file changed since I last ingested it?" - and that one
answer is what makes `/ingest` cheap to re-run and safe to re-run.

  unchanged  ->  skip entirely. No parsing, no embedding, no writes.
  changed    ->  drop the file's old chunks, then add the new ones.
  new        ->  add.

SQLite rather than a JSON file because it is a structured local record with
concurrent-safe writes and needs no server - the same reasoning that chose
Chroma. Its own file, deliberately not shared with observability: Langfuse
replaced the SQLite trace store, but this manifest still fits SQLite well.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from myassistant import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested (
    source      TEXT NOT NULL,
    collection  TEXT NOT NULL,
    hash        TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (source, collection)
)
"""

# Read in blocks so a large PDF does not have to fit in memory twice.
_READ_SIZE = 1 << 20


@contextmanager
def _db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the manifest, creating the table if this is the first run."""
    conn = sqlite3.connect(path or config.MANIFEST_DB)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def file_hash(path: Path) -> str:
    """Hash a file's *contents*.

    Contents, not mtime: copying a file, restoring it from a backup, or letting
    a sync tool touch it all change the timestamp without changing a word, and
    each would trigger a pointless re-embed of the whole document.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            digest.update(block)
    return digest.hexdigest()


def is_unchanged(path: Path, collection: str, db: Path | None = None) -> bool:
    """True when this exact content is already in this collection."""
    with _db(db) as conn:
        row = conn.execute(
            "SELECT hash FROM ingested WHERE source = ? AND collection = ?",
            (str(path), collection),
        ).fetchone()
    return bool(row) and row[0] == file_hash(path)


def record(path: Path, collection: str, db: Path | None = None) -> None:
    """Note that this file's current content is now in this collection.

    REPLACE rather than INSERT: a changed file keeps one row, updated in place.
    The primary key is (source, collection), so the same file legitimately
    appears once per collection it was ingested into.
    """
    with _db(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ingested VALUES (?, ?, ?, ?)",
            (str(path), collection, file_hash(path), datetime.now(UTC).isoformat()),
        )


def forget(path: Path, collection: str, db: Path | None = None) -> None:
    """Drop the record for one file, so the next run treats it as new."""
    with _db(db) as conn:
        conn.execute(
            "DELETE FROM ingested WHERE source = ? AND collection = ?",
            (str(path), collection),
        )


def sources(collection: str, db: Path | None = None) -> list[str]:
    """Every file recorded in a collection - backs `/ingest` reporting."""
    with _db(db) as conn:
        rows = conn.execute(
            "SELECT source FROM ingested WHERE collection = ? ORDER BY source",
            (collection,),
        ).fetchall()
    return [row[0] for row in rows]
