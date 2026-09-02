"""The supervisor - decides which agent answers, and nothing else.

    you ask something
       |
       v
    supervisor (llama3.2:3b)   picks ONE agent, hands off, never answers itself
       |
       +--> research_agent   web + docs. Slower, but grounded in something.
       +--> general_agent    fast, no tools, always tagged UNGROUNDED
       |
       v
    the agent's answer, carrying its confidence tier

Routing is the only job. An agent that both routes and answers is the
combination the one-job rule exists to prevent.

Two agents today. coding_agent (needs tools/safety.py) and docs_agent (needs
the RAG pipeline) join later - deliberately after this routing is proven,
because tuning a four-way router before knowing two-way works makes it
impossible to tell whether a misroute is the model, the agent count, or one
bad description.
"""

from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from myassistant import config
from myassistant.agents import general_agent, research_agent
from myassistant.state import AssistantState

NAME = "supervisor"

# The routing rule that matters for this project's actual use case: a 3B model
# with no tools will fluently answer "what's Spark's broadcast threshold?" and
# be specifically wrong. Naming the cheap-but-ungrounded option last, and
# spelling out what does NOT belong there, is the lever - a small model follows
# a concrete "if it names a technology, use research_agent" far more reliably
# than an abstract instruction to be careful.
PROMPT = f"""You route each question to exactly one agent. You never answer \
questions yourself.

Agents:
- {research_agent.NAME}: searches the web and reads pages. Use it whenever the \
answer depends on a fact that could be looked up - anything naming a specific \
technology, product, service, library, version, configuration setting, error \
message or company. Slower, but its answers are grounded in real sources.
- {general_agent.NAME}: has no tools and cannot check anything. Use it only for \
drafting and rewriting text, general conversation, and explaining a broad \
concept that needs no specific facts.

When unsure, choose {research_agent.NAME}: a slow grounded answer is better \
than a fast unverifiable one.

Hand off to one agent. Do not add commentary of your own."""


def _model() -> ChatOllama:
    """The router model, kept warm longest - it runs on every single turn."""
    return ChatOllama(
        model=config.SUPERVISOR_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.SUPERVISOR_KEEP_ALIVE,
    )


def build(model: Any | None = None, agents: list[Any] | None = None) -> Any:
    """Compile the full graph. `model`/`agents` injectable so tests skip Ollama.

    output_mode="last_message" keeps only the agent's final answer in the
    transcript rather than its whole tool-calling trace. The trace is not lost -
    it is in the Langfuse span - but replaying it into the next turn's context
    would spend a small model's limited window on its own scratch work.
    """
    return create_supervisor(
        agents if agents is not None else [research_agent.build(), general_agent.build()],
        model=model or _model(),
        prompt=PROMPT,
        state_schema=AssistantState,
        supervisor_name=NAME,
        output_mode="last_message",
        # The supervisor decides one thing at a time. Allowing several handoff
        # calls in one turn is where small models produce malformed routing.
        parallel_tool_calls=False,
    ).compile()


__all__ = ["NAME", "PROMPT", "build"]
