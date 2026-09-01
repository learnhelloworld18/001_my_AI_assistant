"""REPL entry point: meta-commands, autocomplete, and error handling.

Meta-commands (anything starting with '/') are handled here and never reach an
agent - faster and more predictable than asking a model to recognise them.

No agent routing yet: answer() returns a placeholder, replaced by the
supervisor call in step 2.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

# Imported first so load_dotenv() runs before anything reads os.environ.
from myassistant import config

log = logging.getLogger("myassistant")


# @dataclass writes __init__/__repr__/__eq__ from the fields below.
# frozen=True blocks mutation - COMMANDS is shared global state.
@dataclass(frozen=True)
class Command:
    """One meta-command: its help text, its handler, and how it completes."""

    help: str  # shown by /help and as autocomplete hint text
    run: Callable[[str, Session], bool]  # (argument, session) -> keep looping?
    takes_path: bool = False  # if True, its argument gets path completion


class Session:
    """In-memory state for one REPL run. Cleared by /clear.

    Deliberately not persisted - cross-session continuity comes from
    conversation_memory in Chroma (step 6), not from raw transcripts.
    """

    def __init__(self) -> None:
        """Start empty, with a fresh id and start time."""
        self.history: list[tuple[str, str]] = []
        # Groups this run's Langfuse traces into one session (step 2/8) - without it each turn shows up as an unrelated trace.
        self.session_id = str(uuid.uuid4())
        # Used by the end-of-session summary (step 6) and /stats (step 9).
        self.started_at = datetime.now(UTC)

    def record(self, user: str, reply: str) -> None:
        """Append one completed exchange to the history."""
        self.history.append((user, reply))

    def clear(self) -> None:
        """Drop all history - backs /clear.

        Keeps session_id and started_at: /clear resets the conversation, it
        does not start a new session.
        """
        self.history.clear()


# --- Meta-commands ---------------------------------------------------------
#
# COMMANDS below is the single source of truth: both the dispatcher and the
# autocomplete read from it, so the menu can never offer something that doesn't
# run, and a new command needs adding in exactly one place.
#
# Every handler takes (arg, session) and returns True to keep looping.


def _cmd_help(arg: str, session: Session) -> bool:
    """Print every registered command and its description."""
    width = max(len(name) for name in COMMANDS)  # align the descriptions
    for name, cmd in sorted(COMMANDS.items()):
        print(f"  {name:<{width}}  {cmd.help}")
    return True


def _cmd_exit(arg: str, session: Session) -> bool:
    """Quit the REPL - the only handler that returns False."""
    return False


def _cmd_clear(arg: str, session: Session) -> bool:
    """Forget this session's history without restarting the process."""
    session.clear()
    print("session cleared")
    return True


def _not_yet(step: int) -> Callable[[str, Session], bool]:
    """Placeholder handler for a command whose feature isn't built yet.

    Registered rather than hidden, so the command still autocompletes and says
    honestly which build step will implement it.
    """

    def run(arg: str, session: Session) -> bool:
        """Say which step implements this, then carry on."""
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

    Plain conversation gets no popup - the common case stays uncluttered, which
    is the point of scoping prompt_toolkit to input only.
    """

    def __init__(self) -> None:
        """Hold a PathCompleter to delegate to for path arguments."""
        self._paths = PathCompleter(expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Yield suggestions for the current input. Called by prompt_toolkit."""
        text = document.text_before_cursor
        if not text.startswith("/"):
            return  # ordinary question - no suggestions

        # No space typed yet => still completing the command name itself.
        name, sep, arg = text.partition(" ")

        if not sep:
            for cmd_name, cmd in sorted(COMMANDS.items()):
                if cmd_name.startswith(name):
                    # Negative start_position replaces what's typed so far.
                    yield Completion(cmd_name, start_position=-len(name), display_meta=cmd.help)
            return

        # Past the space: complete the argument, but only for path commands.
        target = COMMANDS.get(name)
        if target and target.takes_path:
            yield from self._paths.get_completions(Document(arg, len(arg)), complete_event)


# --- Turn handling ---------------------------------------------------------


def handle_meta(line: str, session: Session) -> bool:
    """Dispatch a '/command [arg]' line. Unknown commands are not errors."""
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
    """Handle one line of input. Returns False when the REPL should exit.

    Raises on failure - main() owns the error handling, so tests can assert on
    real exceptions instead of parsing printed output.
    """
    if line.startswith("/"):
        return handle_meta(line, session)

    reply = answer(line, session)
    print(reply)
    session.record(line, reply)
    return True


# --- Entry point -----------------------------------------------------------


def main() -> None:
    """Installed as the `myassistant` command (see [project.scripts])."""
    # Log to ASSISTANT_HOME, not cwd - the app runs from anywhere.
    logging.basicConfig(
        filename=config.ASSISTANT_HOME / "assistant.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    session = Session()
    prompt = PromptSession(
        # History is global, so recall works across launch directories.
        history=FileHistory(str(config.HISTORY_FILE)),
        completer=MetaCommandCompleter(),
        complete_while_typing=True,  # show the menu as you type '/', no Tab needed
    )

    # PROJECT_ROOT is what coding_agent will be fenced to - worth showing.
    print(f"working in {config.PROJECT_ROOT}")
    print("/help for commands, /exit to quit\n")

    while True:
        # Reading input is separate from handling it: Ctrl-C at an empty prompt
        # should clear the line, not be caught by the turn-level handler below.
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
            print("\n(cancelled)")  # Ctrl-C during a slow model call
        except Exception as e:
            # One bad turn must never kill the REPL. Full traceback goes to the
            # log; the user gets one line and their prompt back.
            log.exception("turn failed: %s", line)
            print(f"error: {e}")

    print("bye")


if __name__ == "__main__":
    main()
