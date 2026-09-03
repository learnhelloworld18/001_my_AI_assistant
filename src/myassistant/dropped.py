"""Recognise a file path dragged onto the terminal, and ask before reading it.

    you drag a file onto the terminal
       |
       v
    the shell pastes a SHELL-ESCAPED path:  /Users/him/my\\ diagram.png
       |
       v
    unescape -> does it resolve to a real file?
       |                          |
      yes                         no
       |                          |
    ask permission           treat as an ordinary question
       |
       v
    read it (images -> vision model, text -> plain read)

Two things make this less trivial than it looks.

**The path arrives escaped.** Dragging into a macOS terminal pastes
`/Users/him/my\\ diagram.png` or `'/Users/him/my diagram.png'`, neither of which
is a usable path until unescaped. That is also what makes detection safe: a
sentence almost never unescapes to an existing file.

**A dragged file is almost always outside PROJECT_ROOT** - that is the point of
dragging it - and PROJECT_ROOT is coding_agent's hard fence. So this is a
deliberate, confirmed exception to that fence, and it is narrowed on purpose:
read-only, one file at a time, confirmed each time, and never a file whose name
suggests credentials. Confirmation does not override the denylist; otherwise
drag-and-drop becomes the way around the secrets rule.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from myassistant.tools.read_image import READABLE as IMAGE_SUFFIXES

# Never readable, confirmed or not. A prompt asking "may I read your .env?" is
# not a safety feature - people say yes to prompts. See PROJECT_REQUIREMENTS'
# Safety boundaries.
DENIED_NAMES = {".env", ".netrc", ".npmrc", "credentials", "id_rsa", "id_ed25519"}
DENIED_PATTERNS = ("*.pem", "*.key", "*.p12", "*.keystore", "*credential*", "*secret*", ".env.*")
DENIED_DIRS = {".ssh", ".aws", ".gnupg", ".config/gcloud"}

# Text formats read directly. Anything else falls to the ingest loaders.
TEXT_SUFFIXES = {".md", ".txt", ".py", ".sql", ".json", ".yaml", ".yml", ".toml", ".csv", ".log"}


def as_path(line: str) -> Path | None:
    """The file this line refers to, or None if it is an ordinary question.

    Uses shlex so quoting and backslash-escapes are handled the way the shell
    that produced them intended, rather than by guessing at the escaping.
    """
    text = line.strip()
    if not text:
        return None
    try:
        parts = shlex.split(text)
    except ValueError:  # unbalanced quote - a sentence, not a path
        return None
    if len(parts) != 1:  # a dragged path is exactly one token once unescaped
        return None

    candidate = Path(parts[0]).expanduser()
    # The existence check is the real test. A question does not resolve to a
    # file, so nothing else needs to guess at intent.
    return candidate if candidate.is_file() else None


def is_denied(path: Path) -> bool:
    """Is this file off-limits regardless of confirmation?"""
    if path.name.lower() in DENIED_NAMES:
        return True
    if any(path.match(pattern) for pattern in DENIED_PATTERNS):
        return True
    parts = {p.lower() for p in path.parts}
    return any(d.split("/")[0] in parts for d in DENIED_DIRS)


def kind(path: Path) -> str:
    """How to read this file: "image", "text", or "unsupported"."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES or suffix in {".pdf", ".docx"}:
        return "text"
    return "unsupported"


def describe(path: Path) -> str:
    """The confirmation line. Shows the resolved path and size, so what is
    being agreed to is what will actually be read - not what was typed."""
    size = path.stat().st_size
    unit = f"{size / 1e6:.1f}MB" if size >= 1e6 else f"{size / 1e3:.0f}KB"
    return f"read {kind(path)} file {path.resolve()} ({unit})?"
