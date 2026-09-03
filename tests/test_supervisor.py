"""supervisor: routing only, tested without Ollama.

These assert the wiring and the contract - that a handoff reaches the right
agent and the agent's tier survives back to the caller. Whether llama3.2:3b
*chooses* well is the live checkpoint, not something a mocked test can answer.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from myassistant import supervisor as sup
from myassistant.agents import coding_agent, docs_agent, general_agent, research_agent
from myassistant.state import AssistantState, ConfidenceTier


class _Router(BaseChatModel):
    """Routes to a named agent once, then stops."""

    target: str = ""
    calls: int = 0
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "router"

    def bind_tools(self, tools, **kw):
        self.seen.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls += 1
        if self.calls == 1 and self.target:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": f"transfer_to_{self.target}",
                        "args": {},
                        "id": "h1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            msg = AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _stub_agent(name, text, tier):
    """A stand-in agent that answers and sets a tier, like the real ones do."""

    def answer(state: AssistantState):
        return {"messages": [AIMessage(content=text, name=name)], "confidence": tier}

    g = StateGraph(AssistantState)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile(name=name)


RESEARCH = _stub_agent(research_agent.NAME, "Grounded answer.", ConfidenceTier.HIGH)
GENERAL = _stub_agent(general_agent.NAME, "Quick answer.", ConfidenceTier.UNGROUNDED)


def _run(target, question="what is the broadcast join threshold?"):
    router = _Router(target=target)
    graph = sup.build(model=router, agents=[RESEARCH, GENERAL])
    out = graph.invoke({"messages": [HumanMessage(content=question)]})
    return out, router


def test_a_handoff_reaches_the_research_agent():
    out, _ = _run(research_agent.NAME)
    assert any(getattr(m, "name", None) == research_agent.NAME for m in out["messages"])


def test_a_handoff_reaches_the_general_agent():
    out, _ = _run(general_agent.NAME)
    assert any(getattr(m, "name", None) == general_agent.NAME for m in out["messages"])


def test_the_agents_tier_survives_back_to_the_caller():
    """The tier is the whole point - if it is lost here, nothing can show it."""
    assert _run(research_agent.NAME)[0]["confidence"] is ConfidenceTier.HIGH
    assert _run(general_agent.NAME)[0]["confidence"] is ConfidenceTier.UNGROUNDED


def test_the_supervisor_is_offered_one_handoff_tool_per_agent():
    """This list is the routing decision - it must not silently gain entries."""
    _, router = _run(research_agent.NAME)
    offered = router.seen[0]
    assert any(research_agent.NAME in t for t in offered)
    assert any(general_agent.NAME in t for t in offered)
    assert len(offered) == 2


def test_the_prompt_names_every_agent():
    for agent in (coding_agent, docs_agent, research_agent, general_agent):
        assert agent.NAME in sup.PROMPT


def test_the_prompt_breaks_ties_toward_grounding():
    """A slow checked answer beats a fast unverifiable one - say so explicitly."""
    assert "prefer a grounded agent" in sup.PROMPT


def test_the_prompt_separates_their_facts_from_the_worlds():
    """docs_agent and research_agent both ground answers; the split is whose
    fact it is. Stated as a question the model can actually apply."""
    assert "theirs, or the" in sup.PROMPT
    assert docs_agent.NAME in sup.PROMPT


def test_the_prompt_forbids_the_supervisor_answering():
    """A router that also answers is what the one-job rule exists to prevent."""
    assert "never answer questions yourself" in sup.PROMPT.lower()
