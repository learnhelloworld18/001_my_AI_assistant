"""docs_agent - answers from the user's own documents, and only from them.

    search    the ReAct loop over search_notes / search_resume
       |
       v
    gate      deterministic: did a retrieval clear the relevance threshold?

Same two-node shape as research_agent, and deliberately so - the difference is
only which evidence the gate reads. Here it is the RAG relevance score, which
rag.query has already compared against the threshold and folded into
Observation.ok.

The prompt's real job is stopping the model answering from its own weights when
the notes come back thin. A 3B model asked "what did I do at Capital One?" will
invent a plausible answer if retrieval gives it nothing, and that answer would
be indistinguishable from a real one - which is the failure mode this whole
project is built around.
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
from myassistant.tools.search_notes import search_experience, search_notes, search_resume

NAME = "docs_agent"

RECURSION_LIMIT = 2 * config.MAX_TOOL_STEPS + 2

PROMPT = """You answer questions from the user's own saved documents.

Three places to look:
- search_resume: one company or one project - their CV, interview prep, STAR \
answers. Most detail.
- search_experience: their whole career across every role. Use for "walk me \
through my career", "tell me about yourself", or any question spanning more \
than one job.
- search_notes: their technical notes and reference material.

How to work:
- Search before answering. You have no other source for this.
- Quote or closely paraphrase what you find, and name the file it came from.
- If the results are thin or off-topic, say so plainly: "your notes don't \
really cover that". Do not fill the gap from memory - a plausible invented \
detail about their own career is worse than no answer, because they cannot \
tell it apart from a real one.
- If a search returns [TOOL FAILED], the collection may simply be empty - say \
that rather than guessing at its contents."""


def _model() -> ChatOllama:
    """The general model - reading retrieved prose, not writing code."""
    return ChatOllama(
        model=config.RESEARCH_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SPECIALIST_KEEP_ALIVE,
    )


def _gate(state: AssistantState) -> dict[str, Any]:
    """HIGH when a retrieval cleared the relevance threshold.

    The threshold comparison lives in rag.query, not here: it is the retrieval
    that knows its own score, and the gate's job is only to read the verdict.
    """
    return {"confidence": tier_from_observations(list(state.get("observations", [])))}


def build(model: Any | None = None, tools: list[Any] | None = None) -> Any:
    """Compile the agent. `model`/`tools` injectable so tests skip Ollama."""
    agent = create_react_agent(
        model or _model(),
        tools if tools is not None else [search_resume, search_experience, search_notes],
        prompt=PROMPT,
        state_schema=AssistantState,
        name=NAME,
    )

    def search(state: AssistantState) -> dict[str, Any]:
        """Run the loop under the step cap; a cap hit is a stop, not a crash."""
        try:
            return dict(agent.invoke(state, {"recursion_limit": RECURSION_LIMIT}))
        except GraphRecursionError:
            return {
                "messages": [
                    AIMessage(
                        content="I ran out of steps searching your documents.",
                        name=NAME,
                    )
                ]
            }

    graph = StateGraph(AssistantState)
    graph.add_node("search", search)
    graph.add_node("gate", _gate)
    graph.add_edge(START, "search")
    graph.add_edge("search", "gate")
    graph.add_edge("gate", END)
    return graph.compile(name=NAME)


__all__ = ["NAME", "PROMPT", "RECURSION_LIMIT", "build"]
