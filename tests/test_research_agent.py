"""research_agent, tested in isolation - no Ollama, no network.

The hard rule says each agent node gets a real component test before it is
wired into the supervisor graph. The interesting assertions here are about the
gate: it must reach the right tier from evidence the model never commented on.
"""

from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from pydantic import Field

from myassistant import config
from myassistant.agents import research_agent as ra
from myassistant.state import ConfidenceTier
from myassistant.tools.observation import Observation, emit, failed, fetched

REAL_PAGE = "Spark broadcasts the smaller side of the join. " * 40


class _Scripted(BaseChatModel):
    """Replays a scripted list of AIMessages, one per model call.

    A real BaseChatModel rather than a duck type: create_react_agent calls
    bind_tools() and wraps the result in a Runnable, so a plain object is
    rejected before any test logic runs.
    """

    replies: list = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self  # tool schemas are irrelevant - the replies are fixed

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = (
            self.replies[self.calls]
            if self.calls < len(self.replies)
            else AIMessage(content="done")
        )
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _FakeModel(*replies):
    return _Scripted(replies=list(replies))


def _call(name, tid, **args):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": tid, "type": "tool_call"}]
    )


def _fake_tool(name, obs):
    """A tool that always emits the given Observation."""

    @tool(name)
    def _t(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Fake tool."""
        return emit(obs, tool_call_id)

    return _t


def _run(model, tools):
    graph = ra.build(model=model, tools=tools)
    return graph.invoke({"messages": [HumanMessage(content="how does broadcast join work?")]})


def test_reading_a_page_earns_high():
    page = fetched(REAL_PAGE, source="https://spark.example", kind="page", status=200)
    out = _run(
        _FakeModel(_call("reader", "c1", query="x"), AIMessage(content="Broadcast joins...")),
        [_fake_tool("reader", page)],
    )
    assert out["confidence"] is ConfidenceTier.HIGH


def test_snippets_alone_do_not_earn_high():
    """The model may well sound confident here. The gate does not care."""
    snippets = Observation(ok=True, detail="3 results", content="...", metrics={"kind": "search"})
    out = _run(
        _FakeModel(_call("searcher", "c1", query="x"), AIMessage(content="Broadcast joins...")),
        [_fake_tool("searcher", snippets)],
    )
    assert out["confidence"] is ConfidenceTier.LOW


def test_a_failed_fetch_is_low_even_though_the_model_answered():
    """The exact silent-failure case: a redirect shell, and a confident answer."""
    dead = failed("no usable content", source="https://kafka.example", kind="page", status=200)
    out = _run(
        _FakeModel(_call("reader", "c1", query="x"), AIMessage(content="Kafka replicates...")),
        [_fake_tool("reader", dead)],
    )
    assert out["confidence"] is ConfidenceTier.LOW


def test_answering_with_no_tools_at_all_is_low():
    out = _run(_FakeModel(AIMessage(content="I think it broadcasts.")), [])
    assert out["confidence"] is ConfidenceTier.LOW


def test_observations_accumulate_across_tool_calls():
    """Two calls, both kept - this is the operator.add reducer doing its job."""
    page = fetched(REAL_PAGE, source="https://a.example", kind="page", status=200)
    out = _run(
        _FakeModel(
            _call("reader", "c1", query="x"),
            _call("reader", "c2", query="y"),
            AIMessage(content="done"),
        ),
        [_fake_tool("reader", page)],
    )
    assert len(out["observations"]) == 2


def test_a_looping_model_is_stopped_at_the_cap_without_crashing():
    """The spec: exhausting the cap is not an error.

    A model that never stops calling tools is the realistic failure for a small
    model. remaining_steps stops it gracefully - LangGraph emits its own
    "need more steps" message rather than raising, and the gate still runs.
    """
    page = fetched(REAL_PAGE, source="https://a.example", kind="page", status=200)
    looping = _FakeModel(*[_call("reader", f"c{i}", query="x") for i in range(50)])
    out = _run(looping, [_fake_tool("reader", page)])
    assert len(out["observations"]) == config.MAX_TOOL_STEPS  # the cap bound
    assert "confidence" in out  # the gate still ran
    assert "need more steps" in str(out["messages"][-1].content)


def test_the_step_cap_follows_config():
    """One source of truth - changing MAX_TOOL_STEPS must move the real limit."""
    assert ra.RECURSION_LIMIT == 2 * config.MAX_TOOL_STEPS + 2


def test_the_prompt_tells_the_model_snippets_are_not_evidence():
    """Prompt and gate must agree, or the model is set up to be marked down."""
    assert "never answer from snippets alone" in ra.PROMPT
    assert "[TOOL FAILED]" in ra.PROMPT


def test_the_prompt_forbids_narrating():
    """The agent sees handoff bookkeeping and will mimic it: "the research agent
    has been asked...". Filtering supervisor tokens does not catch that, because
    it is the agent itself talking."""
    assert "Do not narrate" in ra.PROMPT
    assert "third person" in ra.PROMPT
