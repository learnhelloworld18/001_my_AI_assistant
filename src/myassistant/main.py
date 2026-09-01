"""REPL entry point: meta-commands, autocomplete, and error handling.

No agent routing yet - one hardcoded response path, replaced by the supervisor
in step 2.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

from myassistant import config

log = logging.getLogger("myassistant")


@dataclass(frozen=True)
class Command:
    help: str
    run: Callable[[str, Session], bool]  # returns False to exit the REPL
    takes_path: bool = False


class Session:
    """In-memory state for one REPL run. Cleared by /clear."""

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    def record(self, user: str, reply: str) -> None:
        self.turns.append((user, reply))

    def clear(self) -> None:
        self.turns.clear()


# --- Meta-commands ---------------------------------------------------------
# One registry drives both dispatch and autocomplete, so they cannot drift.


def _cmd_help(arg: str, session: Session) -> bool:
    width = max(len(name) for name in COMMANDS)
    for name, cmd in sorted(COMMANDS.items()):
        print(f"  {name:<{width}}  {cmd.help}")
    return True


def _cmd_exit(arg: str, session: Session) -> bool:
    return False


def _cmd_clear(arg: str, session: Session) -> bool:
    session.clear()
    print("session cleared")
    return True


def _not_yet(step: int) -> Callable[[str, Session], bool]:
    def run(arg: str, session: Session) -> bool:
        print(f"not implemented yet (step {step})")
        return True

    return run


COMMANDS: dict[str, Command] = {
    "/help": Command("show this help", _cmd_help),
    "/exit": Command("quit", _cmd_exit),
    "/clear": Command("forget this session's history", _cmd_clear),
    "/ingest": Command("add documents to the knowledge base", _not_yet(4), takes_path=True),
    "/remember": Command("save a note to long-term memory", _not_yet(6)),
    "/stats": Command("recent performance summary", _not_yet(9)),
}


# --- Autocomplete ----------------------------------------------------------


class MetaCommandCompleter(Completer):
    """Completes only lines starting with '/'.

    Plain conversation gets no popup - the common case stays uncluttered.
    """

    def __init__(self) -> None:
        self._paths = PathCompleter(expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        name, sep, arg = text.partition(" ")

        if not sep:
            for cmd_name, cmd in sorted(COMMANDS.items()):
                if cmd_name.startswith(name):
                    yield Completion(cmd_name, start_position=-len(name), display_meta=cmd.help)
            return

        target = COMMANDS.get(name)
        if target and target.takes_path:
            yield from self._paths.get_completions(Document(arg, len(arg)), complete_event)


# --- Turn handling ---------------------------------------------------------


def handle_meta(line: str, session: Session) -> bool:
    name, _, arg = line.partition(" ")
    cmd = COMMANDS.get(name)
    if cmd is None:
        print(f"unknown command {name} - try /help")
        return True
    return cmd.run(arg.strip(), session)


def answer(question: str, session: Session) -> str:
    """Placeholder for the supervisor call added in step 2."""
    return f"(no agents wired up yet) you asked: {question}"


def run_turn(line: str, session: Session) -> bool:
    """Returns False when the REPL should exit."""
    if line.startswith("/"):
        return handle_meta(line, session)

    reply = answer(line, session)
    print(reply)
    session.record(line, reply)
    return True


# --- Entry point -----------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        filename=config.ASSISTANT_HOME / "assistant.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    session = Session()
    prompt = PromptSession(
        history=FileHistory(str(config.HISTORY_FILE)),
        completer=MetaCommandCompleter(),
        complete_while_typing=True,
    )

    print(f"working in {config.PROJECT_ROOT}")
    print("/help for commands, /exit to quit\n")

    while True:
        try:
            line = prompt.prompt("> ").strip()
        except KeyboardInterrupt:
            continue  # Ctrl-C clears the line, it does not quit
        except EOFError:
            break  # Ctrl-D quits

        if not line:
            continue

        try:
            if not run_turn(line, session):
                break
        except KeyboardInterrupt:
            print("\n(cancelled)")
        except Exception as e:
            # One bad turn must never kill the REPL.
            log.exception("turn failed: %s", line)
            print(f"error: {e}")

    print("bye")


if __name__ == "__main__":
    main()
