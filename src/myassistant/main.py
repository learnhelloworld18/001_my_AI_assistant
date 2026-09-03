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
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

# Imported first so load_dotenv() runs before anything reads os.environ.
from myassistant import config, dropped
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


def _cmd_ingest(arg: str, session: Session) -> bool:
    """/ingest <path> [notes|resume] - read documents into a collection.

    Safe to re-run: unchanged files cost nothing and changed ones replace their
    own chunks. Defaults to the notes collection, since that is the one that
    grows; the resume collection is ingested deliberately and rarely.
    """
    from myassistant.rag.ingest import ingest
    from myassistant.rag.store import Collection

    parts = arg.split()
    if not parts:
        print("usage: /ingest <path> [notes|resume]")
        return True

    target = parts[-1].lower() if len(parts) > 1 else "notes"
    path = Path(" ".join(parts[:-1]) if len(parts) > 1 else parts[0]).expanduser()
    collection = (
        Collection.RESUME_INTERVIEW if target.startswith("resume") else Collection.TECH_NOTES
    )

    if not path.exists():
        print(f"no such path: {path}")
        return True

    print(f"reading {path} into {collection} ...")
    result = ingest(path, collection)
    print(result.summary())
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
    "/ingest": Command(
        "add documents to the knowledge base: /ingest <path> [notes|resume]",
        _cmd_ingest,
        takes_path=True,
    ),
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


# Must match supervisor.NAME. Kept as a literal rather than imported, because
# importing that module pulls in langgraph_supervisor and the Ollama client at
# REPL startup; a test asserts the two stay in step.
SUPERVISOR_NAME = "supervisor"


def _speaker(namespace: tuple[str, ...]) -> str | None:
    """Which agent produced this chunk, or None if it was the supervisor.

    LangGraph tags every streamed chunk with the subgraph that produced it -
    ('research_agent:<uuid>',) for an agent, ('supervisor:<uuid>',) or () for
    the supervisor itself. The uuid changes per invocation, so the name before
    the colon is the stable identity.

    That tag does two jobs. It drops the supervisor's commentary - it narrates
    before handing off and paraphrases afterwards - and it identifies repeats,
    below.
    """
    if not namespace:
        return None
    name = namespace[0].split(":", 1)[0]
    return None if name == SUPERVISOR_NAME else name


def _final_text(messages: list[BaseMessage]) -> str:
    """The last thing an *agent* said, falling back to the last message at all.

    Prefers agent messages for the same reason the token stream does: the
    supervisor's closing paraphrase is not the answer, it is a summary of one.
    """
    replies = [m for m in messages if isinstance(m, AIMessage) and str(m.content).strip()]
    for message in reversed(replies):
        if getattr(message, "name", None) not in (None, SUPERVISOR_NAME):
            return str(message.content).strip()
    # No agent spoke. Fall back to the supervisor's own words rather than to
    # nothing - but never to a HumanMessage, which would echo the question back
    # at the user as if it were the answer.
    if replies:
        return str(replies[-1].content).strip()

    # Nothing at all: the supervisor answered with an empty message and never
    # handed off. A small router does this intermittently. Saying so, with
    # something to try, beats "(no answer produced)" - which reads like a bug
    # in the assistant rather than something the user can act on.
    return (
        "I didn't pick an agent for that. Try naming what it's about - a file "
        "or function for your code, 'my notes' for your documents, or ask me "
        "to look it up."
    )


def _tool_calls(update: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tool call in an update, flattened. Empty when there are none."""
    calls: list[dict[str, Any]] = []
    for payload in update.values():
        for message in (payload or {}).get("messages", []) or []:
            calls.extend(getattr(message, "tool_calls", None) or [])
    return calls


def _finished_agents(namespace: tuple[str, ...], update: dict[str, Any]) -> list[str]:
    """Agents that just finished, from a root-level update.

    Root updates are useless as progress lines because they fire only once a
    subgraph has completed - which is exactly what makes them the right signal
    for "this agent is done". Read here rather than from a
    transfer_back_to_supervisor call, because those messages are switched off
    (see supervisor.build) precisely so they do not leak into the answer.
    """
    if namespace:
        return []
    return [node for node in update if node != SUPERVISOR_NAME]


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

    for call in _tool_calls(update):
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
    # Agents that have already answered this turn. A 3B supervisor routinely
    # hands the same question to the same agent twice and prints the answer
    # again; telling it not to in the prompt does not hold. Tracking by agent
    # rather than blocking all repeats keeps genuine chains (research, then
    # coding) working - those are different agents.
    answered: set[str] = set()
    speaking: str | None = None  # the agent whose prose is currently on screen
    # Two separate things, conflated once and it cost an hour: `streamed` is
    # everything printed this turn (and becomes the history entry), while
    # `mid_block` only tracks whether a line is currently open, for spacing.
    # Clearing the first for the second made the end-of-turn fallback think
    # nothing had streamed, so it printed the whole answer a second time.
    mid_block = False

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
            # Mark agents done as they complete, so a re-route to one that has
            # already answered is caught before any repeated token is printed.
            answered.update(_finished_agents(namespace, chunk))
            status = _status(namespace, chunk)

            # Two suppressions, both about re-routing to an agent that is done.
            # `answered` covers one that completed; `speaking` covers the common
            # case - the supervisor tries to hand back to the agent that just
            # finished talking, which would print a heading with nothing under
            # it. Checked here rather than by marking on every update: internal
            # updates arrive *during* streaming, so marking on those cut the
            # answer off mid-sentence.
            if status:
                target = status.removeprefix("· ")
                if target in answered or target == speaking:
                    status = None

            if status and status != last_status:
                if mid_block:
                    # Interrupted mid-answer by real new work (a tool call, a
                    # different agent) - close the line and stop this agent from
                    # repeating itself afterwards.
                    if speaking and not status.endswith("…"):
                        answered.add(speaking)
                    print()
                    mid_block = False
                print(status)
                last_status = status
        else:
            message, _meta = chunk
            speaker = _speaker(namespace)
            # Three filters, each stopping a different duplicate:
            #   speaker is None  the supervisor's own commentary
            #   already answered  the same agent, asked the same thing twice
            #   AIMessageChunk    chunks only; the finished AIMessage is emitted
            #                     too, and printing both would double every reply
            if speaker is None or speaker in answered:
                continue
            if isinstance(message, AIMessageChunk) and message.content:
                text = str(message.content)
                print(text, end="", flush=True)
                streamed.append(text)
                speaking = speaker
                mid_block = True

    if mid_block:
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


def _confirm(question: str) -> bool:
    """Ask a yes/no question, defaulting to no.

    Defaulting to no matters: this gate is the only thing standing between a
    dragged path and a file outside PROJECT_ROOT being read, and a stray Enter
    should decline rather than accept.
    """
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def handle_dropped(path: Path, session: Session) -> bool:
    """A file was dragged onto the terminal. Ask, then read it.

    Deliberately a confirmed exception to coding_agent's working-directory
    fence: a dragged file is almost always outside PROJECT_ROOT, which is the
    point of dragging it. Narrowed to compensate - read-only, one file, asked
    every time, and the denylist is not negotiable by saying yes.
    """
    if dropped.is_denied(path):
        print(f"refusing to read {path.name}: it looks like a credential file")
        return True

    what = dropped.kind(path)
    if what == "unsupported":
        print(f"I can't read {path.suffix} files")
        return True

    if not _confirm(dropped.describe(path)):
        print("skipped")
        return True

    if what == "image":
        from myassistant.tools.read_image import look

        print(f"· reading {path.name} …")
        observation = look(path)
    else:
        from myassistant.rag.ingest import load
        from myassistant.tools.observation import Observation, failed

        try:
            text = load(path)
        except Exception as e:  # noqa: BLE001 - an unreadable file is not a crash
            text, error = "", f"could not read the file: {e}"
        else:
            error = ""

        # Deliberately not fetched(): its threshold asks "is this a real page?",
        # which is the wrong question here. A 300-character note is a perfectly
        # good file, and the user pointed at this one on purpose - the only
        # failure worth reporting is that nothing came out at all.
        if error or not text.strip():
            observation = failed(
                error or "the file has no readable text (scanned or empty?)",
                source=str(path),
                kind="file",
            )
        else:
            observation = Observation(
                ok=True,
                detail=f"read {path.name}",
                content=text,
                source=str(path),
                metrics={"kind": "file", "chars": len(text.strip())},
            )

    print(observation.render())
    # Recorded as an exchange so the next question can refer to "it" - the
    # content is in context without being stored in the vector store, which is
    # the difference between this and /ingest.
    session.record(f"[read {path.name}]", str(observation.content or observation.detail))
    return True


def run_turn(line: str, session: Session) -> bool:
    """Handle one line of input. Returns False when the REPL should exit.

    Raises on failure - main() owns the error handling, so tests can assert on
    real exceptions instead of parsing printed output.
    """
    # Checked before meta-commands, not after: every absolute path on macOS
    # starts with "/", so a dragged file would otherwise be dispatched as an
    # unknown command. Safe in this order because no meta-command resolves to
    # an existing file - "/help" is not a path, and "/ingest <path>" is two
    # tokens, which as_path() rejects.
    if (path := dropped.as_path(line)) is not None:
        return handle_dropped(path, session)

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
