"""REPL entry point: meta-commands, autocomplete, and error handling.

Meta-commands (anything starting with '/') are handled here and never reach an
agent - faster and more predictable than asking a model to recognise them.

answer() hands the turn to the supervisor, which routes it to one agent, and
streams that agent's reply to the screen as it is generated, followed by an
evidence-based confidence tier.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

# Imported first so load_dotenv() runs before anything reads os.environ.
from myassistant import config
from myassistant.observability import langfuse_client
from myassistant.state import ConfidenceTier

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
        # BaseMessage, not (user, reply) tuples: a turn now produces several
        # messages and the graph speaks in messages. Only the question and the
        # final answer are kept - replaying handoffs and tool output would
        # spend a 3B model's context window on its own scratch work.
        self.history: list[BaseMessage] = []
        # Groups this run's Langfuse traces into one session (step 2/8) - without it each turn shows up as an unrelated trace.
        self.session_id = str(uuid.uuid4())
        # Used by the end-of-session summary (step 6) and /stats (step 9).
        self.started_at = datetime.now(UTC)

    def record(self, user: str, reply: str) -> None:
        """Append one completed exchange to the history."""
        self.history.extend([HumanMessage(content=user), AIMessage(content=reply)])

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


# Built once, lazily. Constructing it opens Ollama clients, so doing it at
# import time would make `myassistant --help` need a running Ollama, and would
# make every test that imports this module slow.
_graph: Any | None = None


def _supervisor() -> Any:
    """The compiled supervisor graph, built on first use."""
    global _graph
    if _graph is None:
        from myassistant.supervisor import build

        _graph = build()
    return _graph


# HIGH is not shown: a tag on every answer becomes wallpaper, and the useful
# signal is the exception. LOW and UNGROUNDED are always shown, because both
# mean "do not repeat this without checking".
_TAGS = {
    ConfidenceTier.LOW: "[thin evidence - the sources did not really cover this]",
    ConfidenceTier.UNGROUNDED: "[general knowledge, not verified against any source]",
}


def _final_text(messages: list[BaseMessage]) -> str:
    """The last message that actually said something.

    Deliberately not filtered down to the last *agent* message. The supervisor
    currently adds a paraphrase of its own after an agent answers, which breaks
    the one-job rule - left visible on purpose so it can be judged first-hand
    before being designed around (see DESIGN_DECISIONS).
    """
    for message in reversed(messages):
        text = str(message.content).strip()
        if text:
            return text
    return "(no answer produced)"


def _status(namespace: tuple[str, ...], update: dict[str, Any]) -> str | None:
    """A one-line "what is happening now", or None if this update is not worth saying.

    Two things are worth announcing, and both are the moments the user would
    otherwise be staring at nothing:
      - which agent the supervisor picked
      - which tool is about to run, said *before* it runs

    Everything here is read from *tool calls*, never from node completions.
    A root-level update fires when a subgraph has already finished, so
    announcing the agent from it printed "· research_agent" after the answer
    it was meant to introduce. The handoff call happens before the agent runs,
    which is the whole point of a progress line.
    """
    if not namespace:
        return None  # root updates are completion notices, always too late

    for payload in update.values():
        for message in (payload or {}).get("messages", []) or []:
            for call in getattr(message, "tool_calls", None) or []:
                name = call["name"]
                if name.startswith("transfer_back"):
                    return None  # returning to the supervisor is not news
                if name.startswith("transfer_to_"):
                    return f"· {name.removeprefix('transfer_to_')}"
                return f"· {name}…"
    return None


def answer(question: str, session: Session) -> str:
    """Stream one turn to the screen; return the clean answer text for history.

    Prints as it goes rather than returning a string to print, because the point
    is that the first words appear in a second or two. Total time is unchanged -
    perceived time is not, and that is what the responsiveness priority means.

    The returned text carries no confidence tag: the tag is presentation for the
    human, and feeding "[thin evidence]" back into history as if the assistant
    had said it would nudge later turns.
    """
    state = {"messages": [*session.history, HumanMessage(content=question)]}
    config = {
        # Groups every span of this turn under the session's Langfuse trace.
        "callbacks": langfuse_client.get_callbacks(session.session_id),
        # Supervisor hops plus each agent's own budget. Generous here because
        # the agents cap themselves; this only stops a routing loop.
        "recursion_limit": 25,
    }

    final: dict[str, Any] = {}
    streamed: list[str] = []
    last_status: str | None = None

    # subgraphs=True is what makes this stream at all: agents are compiled
    # subgraphs, and without it only node-level updates cross the boundary -
    # no tokens. Verified the hard way.
    for namespace, mode, chunk in _supervisor().stream(
        state, config, stream_mode=["updates", "messages", "values"], subgraphs=True
    ):
        if mode == "values":
            if not namespace:  # root state only; subgraphs emit their own
                final = chunk
        elif mode == "updates":
            status = _status(namespace, chunk)
            if status and status != last_status:
                if streamed:  # a tool call mid-answer: do not run text together
                    print()
                    streamed.clear()
                print(status)
                last_status = status
        else:
            message, _meta = chunk
            # Chunks only. The finished AIMessage is emitted too, and printing
            # both would show every answer twice.
            if isinstance(message, AIMessageChunk) and message.content:
                text = str(message.content)
                print(text, end="", flush=True)
                streamed.append(text)

    if streamed:
        print()

    text = "".join(streamed).strip() or _final_text(final.get("messages", []))
    if not streamed:
        # Nothing streamed - a model without token support, or an empty run.
        # Print the answer rather than leaving the turn looking like a hang.
        print(text)

    tier = final.get("confidence")
    if tier is not None:
        langfuse_client.score(langfuse_client.CONFIDENCE_SCORE, str(tier))
        tag = _TAGS.get(tier)
        if tag:
            print(tag)
    return text


def run_turn(line: str, session: Session) -> bool:
    """Handle one line of input. Returns False when the REPL should exit.

    Raises on failure - main() owns the error handling, so tests can assert on
    real exceptions instead of parsing printed output.
    """
    if line.startswith("/"):
        return handle_meta(line, session)

    # answer() streams to the screen itself, so nothing is printed here - doing
    # both would show every reply twice.
    reply = answer(line, session)
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

    # Langfuse batches spans on a background thread, so without this the last
    # turn of every session is silently never sent.
    langfuse_client.flush()
    print("bye")


if __name__ == "__main__":
    main()
