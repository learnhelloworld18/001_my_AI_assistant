"""research_agent - answers from the open web, and says how well grounded it is.

Two nodes, on purpose:

    research   the ReAct loop: reason -> act -> observe -> reason
       |       (no fourth "evaluate" step - that IS the reasoning step
       |        running again with the observation now in context)
       v
    gate       deterministic. Reads Observation.ok, never the prose.
               Catches the failures the model could not see.

The gate is separate because the model has no privileged access to whether a
tool really worked. A search that returns snippets and a page fetch that
returns a redirect shell both look like progress from inside the loop.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from myassistant import config
from myassistant.state import AssistantState, ConfidenceTier, tier_from_observations
from myassistant.tools.visit_webpage import visit_webpage
from myassistant.tools.web_search import web_search

NAME = "research_agent"

# One superstep per model call and one per tool batch, plus entry/exit. Set from
# MAX_TOOL_STEPS so the cap has a single source of truth.
RECURSION_LIMIT = 2 * config.MAX_TOOL_STEPS + 2

# Says what the tools are for and, more importantly, what does not count as an
# answer. The snippets-are-not-evidence rule is stated here as well as enforced
# in the gate - the prompt makes the model likely to fetch, the gate makes the
# tier honest when it doesn't.
PROMPT = """You research questions using the web.

How to work:
- web_search finds candidate pages. Its snippets are claims ABOUT pages, not \
the pages themselves - never answer from snippets alone.
- visit_webpage reads a page properly. Do this before answering.
- If a fetch comes back [TOOL FAILED], that page gave you nothing. Try a \
different URL from the search results rather than guessing at its contents.
- You have very few steps. Prefer one good page over three skimmed ones.

Answer in plain prose. State what you actually found, and if the pages did not \
cover something, say so rather than filling the gap from memory."""


def _model() -> ChatOllama:
    """The research model, kept warm only briefly - the supervisor is the hot one."""
    return ChatOllama(
        model=config.RESEARCH_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SPECIALIST_KEEP_ALIVE,
    )


def _gate(state: AssistantState) -> dict[str, Any]:
    """Score the evidence. Deterministic - no model call, no extra latency.

    HIGH needs an ok observation tagged kind="page": a page was actually read.
    Search-only turns are LOW, which is the spec's rule, not a heuristic.
    """
    return {"confidence": tier_from_observations(list(state.get("observations", [])))}


def build(model: Any | None = None, tools: list[Any] | None = None) -> Any:
    """Compile the agent. `model`/`tools` are injectable so tests never call Ollama."""
    agent = create_react_agent(
        model or _model(),
        tools if tools is not None else [web_search, visit_webpage],
        prompt=PROMPT,
        state_schema=AssistantState,
        name=NAME,
    )

    def research(state: AssistantState) -> dict[str, Any]:
        """Run the ReAct loop under the step cap.

        The cap is enforced by `remaining_steps` in the state, which LangGraph
        derives from recursion_limit: a model that keeps calling tools is
        stopped after MAX_TOOL_STEPS with LangGraph's own "need more steps"
        message. That is a stop, not a crash, which is what the spec wants -
        the gate still runs and the answer still goes out with a tier.

        The except below is a backstop for paths that bypass that budget. It
        does not fire in the normal loop-forever case (verified in tests).
        """
        try:
            return dict(agent.invoke(state, {"recursion_limit": RECURSION_LIMIT}))
        except GraphRecursionError:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I ran out of research steps before finishing. Here is what "
                            "I gathered, but treat it as incomplete."
                        ),
                        name=NAME,
                    )
                ]
            }

    graph = StateGraph(AssistantState)
    graph.add_node("research", research)
    graph.add_node("gate", _gate)
    graph.add_edge(START, "research")
    graph.add_edge("research", "gate")
    graph.add_edge("gate", END)
    return graph.compile(name=NAME)


__all__ = ["NAME", "PROMPT", "RECURSION_LIMIT", "ConfidenceTier", "build"]
