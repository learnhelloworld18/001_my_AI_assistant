"""coding_agent - reads your project, and writes code.

    read    qwen2.5:3b + tools     gathers files. Produces no prose.
      |
      v
    write   qwen2.5-coder:7b       writes the answer. Has no tools.
      |
      v
    gate    deterministic          was a real file actually read?

Two models, because the two jobs need opposite things and no single local model
here does both.

**Why read cannot use the coder model.** qwen2.5-coder emits its tool calls as
plain text - '{"name": "read_project_file", "arguments": {...}}' - rather than
as structured calls, 0 times out of 4. LangGraph never executes them, so an
agent built on it announces it cannot read files while holding tools that can.

**Why write should not use the 3B.** Generating a window function or a
non-trivial algorithm is what a coder-tuned model is for, and the tool-call
defect is irrelevant here: the write node binds no tools, so there is nothing
to mis-format.

The split also satisfies the rule that no agent both researches and writes its
own final answer. `read` returns evidence and nothing else - its prose is
discarded on purpose - and `write` is the only node that speaks.

**Writes are proposed, never performed.** propose_write and propose_command
queue an action and describe it; main.py shows it, asks, and only then acts.
LangGraph's interrupt() would be the tidier mechanism and was measured: it
works from a tool up through an agent's own graph, but does not survive
langgraph-supervisor's handoff, so the pause never reaches the REPL. Read-only
commands still run immediately - friction only where it matters.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from myassistant import config
from myassistant.state import (
    AssistantState,
    Pending,
    render_evidence,
    tier_from_observations,
)
from myassistant.tools.coding import build_tools
from myassistant.tools.observation import Observation

NAME = "coding_agent"

RECURSION_LIMIT = 2 * config.MAX_TOOL_STEPS + 2

# The reader is told to gather and stop. Its prose is thrown away, so any effort
# spent composing an answer is wasted latency.
READ_PROMPT = """You find and read files in the user's current project.

- list_project_files finds what exists, with a glob like "src/**/*.py".
- read_project_file reads one file.
- Paths are relative to the project root. Anything outside it is refused, and \
so are credential files - that is a boundary, not an error to work around. If \
a read is refused, stop; do not try another path to reach the same file.
- If the question needs no file - "write a SQL query that...", "how do I \
reverse a linked list" - call nothing at all and reply with the single word \
NONE.

If asked to create or change a file, or to run something, use propose_write or \
propose_command. They do not act: they queue the action for the user to \
approve. Read-only commands run straight away.

Do not explain or answer the question. Another step does that. Read what is \
needed and stop."""

# The writer never sees a tool, only the question and whatever was read.
WRITE_PROMPT = """You are a senior engineer answering a coding question.

- Write code in fenced blocks, with the language tagged.
- If file contents are given below, base your answer on what they actually \
say, and name the file you are describing. If none are given, answer from your \
own knowledge - that is expected for a general question.
- Be direct. No preamble about what you are about to do, and no narration of \
handoffs or tools; that is plumbing, not part of the conversation.
- If a write or a command was proposed, say it is waiting for the user's \
approval. Never say you have created, changed or run anything - you have not; \
the user decides after you finish."""


def _reader() -> ChatOllama:
    """The model that calls tools. Small, and the only one here that reliably can."""
    return ChatOllama(
        model=config.SUPERVISOR_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SPECIALIST_KEEP_ALIVE,
    )


def _writer() -> ChatOllama:
    """The coder model. Bound to no tools, which is why it works here."""
    return ChatOllama(
        model=config.CODING_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SPECIALIST_KEEP_ALIVE,
    )


def _question(state: AssistantState) -> str:
    """The last thing the user actually asked, ignoring handoff bookkeeping."""
    for message in reversed(list(state.get("messages", []))):
        if isinstance(message, HumanMessage) and str(message.content).strip():
            return str(message.content)
    return ""


def _gate(state: AssistantState) -> dict[str, Any]:
    """HIGH when a real file was read.

    A general "write me a SQL query" earns no grounding, and should not:
    nothing was checked against anything.
    """
    return {"confidence": tier_from_observations(list(state.get("observations", [])))}


def build(
    reader: Any | None = None, writer: Any | None = None, tools: list[Any] | None = None
) -> Any:
    """Compile the agent. Both models and the tools are injectable for tests."""
    agent = create_react_agent(
        reader or _reader(),
        build_tools() if tools is None else tools,
        prompt=READ_PROMPT,
        state_schema=AssistantState,
        name=f"{NAME}_reader",
    )
    write_model = writer or _writer()

    def read(state: AssistantState) -> dict[str, Any]:
        """Gather files. Returns only new evidence - never prose.

        Discarding the reader's own answer is what keeps this node to one job.
        Letting both nodes speak would show the user two answers, weaker first.
        """
        had_observations = len(list(state.get("observations", [])))
        had_pending = len(list(state.get("pending", [])))
        try:
            result = dict(agent.invoke(state, {"recursion_limit": RECURSION_LIMIT}))
        except GraphRecursionError:
            return {}
        # Only what this node added: the sub-agent returns the whole accumulated
        # list, and both keys have add reducers, so returning all of it would
        # double every entry. `pending` has to be forwarded as carefully as
        # `observations` - dropping it silently swallowed every proposed write,
        # and the agent announced a proposal nobody was ever asked about.
        observations: list[Observation] = list(result.get("observations", []))
        pending: list[Pending] = list(result.get("pending", []))
        return {
            "observations": observations[had_observations:],
            "pending": pending[had_pending:],
        }

    def write(state: AssistantState) -> dict[str, Any]:
        """Answer the question, using whatever was read."""
        observations = list(state.get("observations", []))
        prompt = _question(state)
        if observations:
            prompt = f"{prompt}\n\nFile contents that were read:\n{render_evidence(observations)}"

        reply = write_model.invoke(
            [SystemMessage(content=WRITE_PROMPT), HumanMessage(content=prompt)]
        )
        return {"messages": [AIMessage(content=reply.content, name=NAME)]}

    graph = StateGraph(AssistantState)
    graph.add_node("read", read)
    graph.add_node("write", write)
    graph.add_node("gate", _gate)
    graph.add_edge(START, "read")
    graph.add_edge("read", "write")
    graph.add_edge("write", "gate")
    graph.add_edge("gate", END)
    return graph.compile(name=NAME)


__all__ = ["NAME", "READ_PROMPT", "RECURSION_LIMIT", "WRITE_PROMPT", "build"]
