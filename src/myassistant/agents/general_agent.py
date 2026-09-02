"""general_agent - quick chat, drafting, rephrasing. No tools, no grounding.

    answer   one model call, no loop
       |     (no tools means nothing to observe, so no reason -> act -> observe)
       v
    always ConfidenceTier.UNGROUNDED

Deliberately the only agent with a *constant* tier. The others earn theirs from
evidence; this one has none available, so saying so is the honest signal rather
than a placeholder. That tag is always shown, because it is always true.

The risk this agent carries is specific: a 3B model will fluently answer a
question about Kafka internals or a Spark config default and be specifically
wrong - wrong defaults, wrong key names, behaviour from three versions ago.
Nothing fails, so nothing is noticeable. The prompt below and the permanent
UNGROUNDED tag are the two mitigations; routing technical questions elsewhere
is the third and belongs to the supervisor.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from myassistant import config
from myassistant.state import AssistantState, ConfidenceTier

NAME = "general_agent"

# Says plainly what this agent cannot do, and what to do about it. "Say you
# cannot verify it" is a cheaper instruction to follow than "be accurate", and
# it is the one failure mode that actually matters here: a confidently stated
# wrong specific is worse than no answer, especially for interview prep.
PROMPT = """You handle quick conversational questions, drafting, rephrasing and \
short explanations.

You have no tools: no web access, no search, no access to the user's notes. \
So do not invent specifics. If an answer would need a version number, a \
configuration key, an API signature, a default value, a benchmark figure or a \
date, say you cannot verify it instead of producing a plausible one. A wrong \
specific stated confidently is worse than no answer.

Explaining a concept in general terms is fine and is what you are for. Stating \
what a particular tool does in a particular version is not.

Keep answers short unless asked otherwise."""


def _model() -> ChatOllama:
    """The supervisor's model, which is already warm - see config.GENERAL_MODEL."""
    return ChatOllama(
        model=config.GENERAL_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SUPERVISOR_KEEP_ALIVE,
    )


def build(model: Any | None = None) -> Any:
    """Compile the agent. `model` is injectable so tests never call Ollama.

    Not built with create_react_agent: with no tools there is no loop to run,
    and going through the ReAct machinery would add tool-binding overhead to
    the one agent whose entire purpose is being fast.
    """
    llm = model or _model()

    def answer(state: AssistantState) -> dict[str, Any]:
        """One model call, one message back, one constant tier."""
        reply = llm.invoke([SystemMessage(content=PROMPT), *state["messages"]])
        # Named so the supervisor (and the transcript) can attribute the answer;
        # without it every agent's output looks the same downstream.
        return {
            "messages": [AIMessage(content=reply.content, name=NAME)],
            "confidence": ConfidenceTier.UNGROUNDED,
        }

    graph = StateGraph(AssistantState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile(name=NAME)


__all__ = ["NAME", "PROMPT", "build"]
