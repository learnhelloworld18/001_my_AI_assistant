"""coding_agent - reads the project and writes code about it.

    read      the ReAct loop over list_project_files / read_project_file
       |
       v
    gate      deterministic: was a real file actually read?

The only agent using the 7B coder model, and the only one whose tools touch
real state - which is why tools/safety.py exists and why this agent is the last
one built rather than the first.

**What it can and cannot do, deliberately.** It reads files and lists them. It
does not yet write files or run commands, even though tools/coding.py has the
machinery (plan_write/do_write, plan_shell/do_shell) and safety.py has vetted
it. The missing piece is not safety, it is control flow: the confirmation gate
has to be answered by a human in the REPL, and an agent cannot pause mid-loop
to ask. Wiring that needs an interrupt in the graph, and shipping the read-only
half first means the fence gets exercised on every call before anything can
change a file.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from myassistant import config
from myassistant.state import AssistantState, tier_from_observations
from myassistant.tools.coding import build_tools

NAME = "coding_agent"

RECURSION_LIMIT = 2 * config.MAX_TOOL_STEPS + 2

PROMPT = """You answer questions about the code in the user's current project, \
and write code when asked.

Tools:
- list_project_files: find what exists, with a glob like "src/**/*.py".
- read_project_file: read one file. Do this before describing what code does.

How to work:
- Look before you answer. Guessing at what a file contains, when you could \
read it, is the one thing to avoid here.
- Paths are relative to the project root. Anything outside it is refused, and \
so are credential files - that is a boundary, not an error to work around. If \
a read is refused, say so; do not try another path to get at the same file.
- When you write code, write it in a fenced block and say which file it belongs \
in. You cannot create or modify files yet, so do not claim to have done so.
- If a file is truncated, say which part you saw."""


def _model() -> ChatOllama:
    """The coder model - the one place a 7B is worth its memory."""
    return ChatOllama(
        model=config.CODING_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SPECIALIST_KEEP_ALIVE,
    )


def _gate(state: AssistantState) -> dict[str, Any]:
    """HIGH when a real file was read.

    validate_code will fold into this later, the same way rag.query does: the
    tool decides its own standard and reports it in Observation.ok, and the
    gate only reads the verdict.
    """
    return {"confidence": tier_from_observations(list(state.get("observations", [])))}


def build(model: Any | None = None, tools: list[Any] | None = None) -> Any:
    """Compile the agent. `model`/`tools` injectable so tests skip Ollama."""
    agent = create_react_agent(
        model or _model(),
        build_tools() if tools is None else tools,
        prompt=PROMPT,
        state_schema=AssistantState,
        name=NAME,
    )

    def read(state: AssistantState) -> dict[str, Any]:
        """Run the loop under the step cap; a cap hit is a stop, not a crash."""
        try:
            return dict(agent.invoke(state, {"recursion_limit": RECURSION_LIMIT}))
        except GraphRecursionError:
            return {
                "messages": [
                    AIMessage(content="I ran out of steps reading the project.", name=NAME)
                ]
            }

    graph = StateGraph(AssistantState)
    graph.add_node("read", read)
    graph.add_node("gate", _gate)
    graph.add_edge(START, "read")
    graph.add_edge("read", "gate")
    graph.add_edge("gate", END)
    return graph.compile(name=NAME)


__all__ = ["NAME", "PROMPT", "RECURSION_LIMIT", "build"]
