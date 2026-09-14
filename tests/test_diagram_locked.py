"""v1 of the architecture diagram is frozen.

`make_drawio.py` is done. Changes go to a v2 generator, so this file exists to
make "locked" checkable rather than merely intended: it regenerates v1 and
compares the bytes against the committed artifact.

Why bytes and not "does it still look right": the generator has a lot of
machinery that produces a diagram from a layout - routing, A*, label placement,
line separation - and any of it can be perturbed by an unrelated change to a
shared helper. A checksum notices. An eye, two weeks later, does not.

When v2 legitimately changes shared code, this test is the thing that says so.
If it fails, either the change belongs only in v2, or v1 is being unfrozen
deliberately and this expectation is updated in the same commit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "project_docs" / "make_drawio.py"
ARTIFACT = ROOT / "architecture.drawio"

# sha256 of to_drawio(compose()) + "\n", i.e. exactly what main() writes.
V1_SHA = "66b62546230a5353650c6bb9f7981e139e8455968e3402f122d65e4932c55ba4"


def _load():
    """Import the generator by path - project_docs/ is not a package."""
    spec = importlib.util.spec_from_file_location("make_drawio_v1", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_drawio_v1"] = module
    spec.loader.exec_module(module)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_v1_regenerates_byte_for_byte():
    """The frozen diagram is reproducible from its frozen generator."""
    m = _load()
    assert _sha(m.to_drawio(m.compose()) + "\n") == V1_SHA


def test_the_committed_artifact_matches_the_generator():
    """Guards the other direction: someone editing architecture.drawio by hand
    in draw.io and committing it, which would silently detach the picture from
    the code that is supposed to produce it."""
    assert _sha(ARTIFACT.read_text()) == V1_SHA


@pytest.mark.parametrize(
    "check",
    ["assert_no_overlap", "assert_clusters_clean", "assert_every_edge_labelled"],
)
def test_v1_still_satisfies_its_own_invariants(check):
    """Frozen output is not the same as correct output - the checksum would
    happily freeze a broken diagram. These are the properties v1 claims."""
    m = _load()
    getattr(m.compose(), check)()


def test_v1_routes_are_clear():
    m = _load()
    m.assert_routes_clear(m.compose())
