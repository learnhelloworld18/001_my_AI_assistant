"""coding_agent's tools - the only ones that touch real state.

Every one of them goes through tools/safety.py first. That is not a convention
to remember; it is why these functions are short. The interesting decisions
live in safety.py, and each tool here just asks it and reports the answer.

The confirmation gate is deliberately *not* here. These functions decide what
would happen and say so; main.py asks the user. Keeping the question in the
REPL means a tool cannot accidentally run something unattended, and it means
these stay testable without stubbing input().
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

from myassistant import config
from myassistant.state import Pending
from myassistant.tools.observation import Observation, emit, failed
from myassistant.tools.safety import Unsafe, Verdict, check_command, safe_path

# A shell command that has not finished in this long is stuck, and an
# interactive REPL cannot wait for it.
TIMEOUT_S = 60

# Enough to see what a file contains without filling a 3B context with one file.
MAX_READ_CHARS = 20_000


def read_file(path: str) -> Observation:
    """Read one file inside the working directory."""
    try:
        resolved = safe_path(path, must_exist=True)
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except Unsafe as e:
        return failed(str(e), source=path, kind="file")
    except OSError as e:
        return failed(f"could not read the file: {e}", source=path, kind="file")

    truncated = len(text) > MAX_READ_CHARS
    body = text[:MAX_READ_CHARS] + ("\n...[truncated]" if truncated else "")
    return Observation(
        ok=True,
        detail=f"read {resolved.name}",
        content=body,
        source=str(resolved),
        metrics={"kind": "file", "chars": len(text), "truncated": truncated},
    )


def list_files(pattern: str = "*") -> Observation:
    """List files under the working directory matching a glob."""
    try:
        root = config.PROJECT_ROOT
        matches = sorted(
            p.relative_to(root)
            for p in root.glob(pattern)
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        )
    except (OSError, ValueError) as e:
        return failed(f"could not list files: {e}", kind="file")

    if not matches:
        # "No matches" alone reads as a settled answer; naming the pattern and
        # the root tells the model what to try differently.
        return failed(
            f"nothing under {root} matches {pattern!r}", source=str(root), kind="file", n_results=0
        )
    listing = "\n".join(str(m) for m in matches[:200])
    return Observation(
        ok=True,
        detail=f"{len(matches)} files matching {pattern!r}",
        content=listing,
        source=str(root),
        metrics={"kind": "file", "n_results": len(matches)},
    )


def plan_write(path: str, content: str) -> tuple[Observation | None, Path | None]:
    """Check a proposed write. Returns (refusal, resolved path).

    Split from the write itself so main.py can show the user what will happen
    and get a yes before anything is touched. A tool that both asks and acts
    would have no point at which the answer could be no.
    """
    try:
        return None, safe_path(path)
    except Unsafe as e:
        return failed(str(e), source=path, kind="file"), None


def do_write(resolved: Path, content: str) -> Observation:
    """Write a file whose path has already been checked by plan_write."""
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved.exists()
        resolved.write_text(content, encoding="utf-8")
    except OSError as e:
        return failed(f"could not write the file: {e}", source=str(resolved), kind="file")
    return Observation(
        ok=True,
        detail=f"{'overwrote' if existed else 'created'} {resolved.name}",
        source=str(resolved),
        metrics={"kind": "file", "chars": len(content), "created": not existed},
    )


def plan_shell(command: str) -> tuple[Verdict, str]:
    """What would happen if this command ran. See safety.check_command."""
    return check_command(command)


def do_shell(command: str) -> Observation:
    """Run a command whose verdict has already been checked.

    Always from PROJECT_ROOT, never from wherever the process happens to be:
    a relative path in a command should mean the same thing as a relative path
    given to safe_path.
    """
    try:
        finished = subprocess.run(
            command,
            # shell=True is safe only because check_command has already
            # vetted this string; nothing reaches here unvetted.
            shell=True,
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            # A non-zero exit is a result to report, not an exception - the
            # model needs the error text to decide what to do next.
            check=False,
        )
    except subprocess.TimeoutExpired:
        return failed(f"command timed out after {TIMEOUT_S}s", source=command, kind="shell")
    except OSError as e:
        return failed(f"could not run the command: {e}", source=command, kind="shell")

    output = (finished.stdout + finished.stderr).strip()
    if finished.returncode != 0:
        # A non-zero exit is a real result, not an exception - the model needs
        # to see the error text to decide what to do next.
        return failed(
            f"exited {finished.returncode}: {output[:2000] or '(no output)'}",
            source=command,
            kind="shell",
            exit_code=finished.returncode,
        )
    return Observation(
        ok=True,
        detail=f"ran {command}",
        content=output[:MAX_READ_CHARS] or "(no output)",
        source=command,
        metrics={"kind": "shell", "exit_code": 0},
    )


# --- LangChain tools --------------------------------------------------------
#
# The write and shell tools *propose*; they never act. What they return to the
# model is a description of what would happen, and the action itself is queued
# in state for main.py to confirm with a human. See state.Pending for why the
# tidier mechanism (LangGraph interrupt) could not be used.


@tool
def read_project_file(path: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Read a file from the current project directory.

    Use to see what code actually says before answering about it. Paths are
    relative to the project you launched from; files outside it cannot be read.

    Args:
        path: path relative to the project root, e.g. "src/main.py"
    """
    return emit(read_file(path), tool_call_id)


@tool
def list_project_files(pattern: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """List files in the current project matching a glob pattern.

    Use to find out what exists before reading. Examples: "*.py", "src/**/*.py",
    "tests/test_*.py".

    Args:
        pattern: a glob, relative to the project root
    """
    return emit(list_files(pattern), tool_call_id)


@tool
def propose_write(
    path: str, content: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Propose writing a file. The user is asked before anything is written.

    Use when asked to create or change a file in the project. Give the complete
    intended contents - the file is replaced, not appended to. Nothing happens
    until the user agrees, so say that you have proposed it, never that you
    have done it.

    Args:
        path: path relative to the project root, e.g. "src/new.py"
        content: the complete file contents
    """
    refusal, resolved = plan_write(path, content)
    if refusal is not None or resolved is None:
        # A refusal is final, and never becomes a question. The denylist is not
        # negotiable by saying yes, so it must not reach a prompt at all.
        return emit(refusal or failed("refused", source=path, kind="file"), tool_call_id)

    action = Pending(kind="write", target=str(resolved), content=content)
    # Pauses the whole graph here and hands the question to the REPL. Execution
    # resumes on this line with whatever the user answered, so the agent sees
    # the result of the write inside the same turn.
    if not interrupt({"ask": action.describe(), "preview": action.content[:800]}):
        return emit(
            failed("the user declined that write", source=str(resolved), kind="file"), tool_call_id
        )
    return emit(do_write(resolved, content), tool_call_id)


@tool
def propose_command(command: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Propose running a shell command in the project directory.

    Read-only commands (ls, cat, git status, pytest --collect-only) run without
    asking. Anything that changes state is queued for the user's approval, and
    some commands are refused outright however the request is phrased.

    Args:
        command: the command line to run, from the project root
    """
    verdict, why = plan_shell(command)

    if verdict is Verdict.DENY:
        return emit(
            failed(f"refusing to run that: {why}", source=command, kind="shell"), tool_call_id
        )

    if verdict is Verdict.ALLOW:
        # Provably read-only, so it runs without asking - friction only where
        # it matters.
        return emit(do_shell(command), tool_call_id)

    action = Pending(kind="shell", target=command)
    if not interrupt({"ask": f"{action.describe()} ({why})", "preview": ""}):
        return emit(
            failed("the user declined that command", source=command, kind="shell"), tool_call_id
        )
    return emit(do_shell(command), tool_call_id)


def build_tools() -> list[Any]:
    """The tools coding_agent gets."""
    return [list_project_files, read_project_file, propose_write, propose_command]
