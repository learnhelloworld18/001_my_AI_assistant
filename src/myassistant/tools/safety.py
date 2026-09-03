"""The boundary around coding_agent - the only agent that can change your machine.

Three layers, in order of strictness:

    DENY      never runs, confirmation or not
              sudo, recursive deletes, piping the internet into a shell,
              reading credential files
    CONFIRM   shown to you and run only if you say yes
              anything that writes, deletes, commits, pushes, or that we
              cannot prove is read-only
    ALLOW     runs immediately
              a short list of provably read-only commands

Two principles decide everything here.

**Unknown means confirm, never allow.** The allowlist is the exception and it
is short. A command we do not recognise is treated as dangerous, because the
alternative is a model inventing a command we never considered and it running
unseen.

**A confirmation is not a safety boundary.** People say yes to prompts,
especially the tenth one in a row. So the denylist is not overridable: there is
no path through this module where answering yes reads your `.env` or runs
`sudo`. That is why DENY exists as a separate layer rather than as a scarier
confirmation message.
"""

from __future__ import annotations

import re
import shlex
from enum import StrEnum
from pathlib import Path

from myassistant import config


class Verdict(StrEnum):
    """What may be done with a proposed action."""

    ALLOW = "allow"  # provably read-only
    CONFIRM = "confirm"  # show it, run only on an explicit yes
    DENY = "deny"  # never, whatever the user says


class Unsafe(Exception):
    """A path or command that must not be used. Carries the reason to show."""


# --- paths ------------------------------------------------------------------

# Never read, never write, confirmation or not. Names first, then patterns,
# then whole directories - a file can be dangerous by any of the three.
DENIED_NAMES = {".env", ".netrc", ".npmrc", ".pgpass", "id_rsa", "id_ed25519", "credentials"}
DENIED_PATTERNS = ("*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", ".env.*", "*_rsa", "*_ed25519")
DENIED_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}


def _is_denied_path(path: Path) -> str | None:
    """Why this file is off-limits, or None. Checked on reads as well as writes.

    Reads matter as much as writes here: the risk is not only corrupting a
    credential file but quoting one into a model's context, from where it can
    end up in a log, a trace, or an answer.
    """
    name = path.name.lower()
    if name in DENIED_NAMES:
        return f"{path.name} is a credential file"
    if any(path.match(pattern) for pattern in DENIED_PATTERNS):
        return f"{path.name} looks like a key or credential file"
    lowered = {part.lower() for part in path.parts}
    if hit := lowered & DENIED_DIRS:
        return f"{next(iter(hit))} holds credentials"
    return None


def safe_path(raw: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path and prove it is inside PROJECT_ROOT. Raise Unsafe if not.

    resolve() runs *before* the containment check, and that order is the whole
    guarantee: "../../.ssh/id_rsa" and a symlink pointing at /etc both look
    innocent until resolved, and both land outside the root once they are.

    PROJECT_ROOT is captured once at import (see config), so this fence cannot
    drift if a shell command or a test changes the working directory.
    """
    candidate = Path(raw).expanduser()
    resolved = (
        (config.PROJECT_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )

    if not resolved.is_relative_to(config.PROJECT_ROOT):
        raise Unsafe(f"{resolved} is outside the working directory ({config.PROJECT_ROOT})")
    if reason := _is_denied_path(resolved):
        raise Unsafe(f"refusing to touch {resolved.name}: {reason}")
    if must_exist and not resolved.exists():
        raise Unsafe(f"no such file: {resolved}")
    return resolved


# --- commands ---------------------------------------------------------------

# Provably read-only. Anything not here needs confirmation - the list is
# deliberately short, and grows only when something is checked and found safe.
READ_ONLY = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "file",
    "stat",
    "pwd",
    "which",
    "type",
    "grep",
    "rg",
    "find",
    "fd",
    "diff",
    "tree",
    "du",
    "df",
    "date",
    "echo",
}

# git is read-only for some subcommands and not others, so it is judged on its
# second word rather than its first.
GIT_READ_ONLY = {"status", "diff", "log", "show", "branch", "remote", "blame", "describe"}

# Never run. Each entry is (pattern, why) - the reason is shown to the user, so
# a refusal explains itself instead of just failing.
DENIED_COMMANDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsudo\b|\bsu\b|\bdoas\b"), "privilege escalation"),
    (re.compile(r"\b(mkfs|fdisk|dd)\b"), "raw disk operation"),
    (re.compile(r"\bchmod\s+(-\w+\s+)*777\b"), "world-writable permissions"),
    (re.compile(r"(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|)sh"), "piping a download into a shell"),
    (re.compile(r":\(\)\s*\{.*\}\s*;?\s*:"), "fork bomb"),
    (re.compile(r"\bhistory\s+-c|\bshred\b"), "destroying evidence of what ran"),
)

# Where one command ends and the next begins. A command line is judged segment
# by segment: "ls && rm -rf ~" must not be called read-only because it starts
# with ls.
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n]")


def _denied_words(words: list[str]) -> str | None:
    """Structural denials, judged on parsed words rather than on the raw text.

    `rm` is the case that proves the point: "-rf", "-r -f", "-fr" and
    "--recursive --force" are the same command, and a regex that catches three
    of them is a false sense of safety. Flags are collected and inspected
    instead.
    """
    if not words or Path(words[0]).name != "rm":
        return None

    flags = set()
    targets = []
    for word in words[1:]:
        if word == "--recursive":
            flags.add("r")
        elif word == "--force":
            flags.add("f")
        elif word.startswith("-") and not word.startswith("--"):
            flags.update(word[1:].lower())
        elif not word.startswith("-"):
            targets.append(word)

    if "r" in flags:
        return "recursive delete"
    if any(t in {"/", "~", "/*", "~/*", ".."} or t.startswith("/ ") for t in targets):
        return "delete at the filesystem root"
    return None


def _segment_verdict(segment: str) -> Verdict:
    """Classify one command, with no separators in it."""
    try:
        words = shlex.split(segment)
    except ValueError:  # unbalanced quotes - we cannot read it, so do not trust it
        return Verdict.CONFIRM
    if not words:
        return Verdict.ALLOW

    # Skip leading environment assignments: FOO=1 ls is still ls.
    while words and "=" in words[0] and not words[0].startswith("-"):
        words = words[1:]
    if not words:
        return Verdict.CONFIRM

    name = Path(words[0]).name
    if name == "git":
        subcommand = next((w for w in words[1:] if not w.startswith("-")), "")
        return Verdict.ALLOW if subcommand in GIT_READ_ONLY else Verdict.CONFIRM
    return Verdict.ALLOW if name in READ_ONLY else Verdict.CONFIRM


def check_command(command: str) -> tuple[Verdict, str]:
    """Judge a whole command line. Returns (verdict, reason).

    The denylist is checked against the raw line first, before any splitting:
    an evasion that hides in quoting or spacing should still be caught by a
    pattern that looks at the text as written.

    Redirection is treated as a write even when the command is read-only,
    because `cat x > y` is a write however it is spelled.
    """
    line = command.strip()
    if not line:
        return Verdict.DENY, "empty command"

    for pattern, why in DENIED_COMMANDS:
        if pattern.search(line):
            return Verdict.DENY, why

    # Structural denials, per segment, on parsed words rather than raw text.
    for segment in _SEPARATORS.split(line):
        try:
            words = shlex.split(segment)
        except ValueError:
            continue  # unparseable; the CONFIRM default below still covers it
        if reason := _denied_words(words):
            return Verdict.DENY, reason

    if re.search(r"(?<![0-9<>])>{1,2}(?!&)", line):
        return Verdict.CONFIRM, "writes to a file"

    # The strictest segment wins: one dangerous step makes the whole line so.
    for segment in _SEPARATORS.split(line):
        if _segment_verdict(segment) is Verdict.CONFIRM:
            return Verdict.CONFIRM, "not on the read-only list"
    return Verdict.ALLOW, "read-only"
