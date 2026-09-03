"""docs_agent, tested in isolation - no Ollama, no vector store.

The one that matters: when retrieval comes back thin, the tier must say so even
though the model sounded certain. A 3B asked about someone's own career will
invent a plausible answer, and they cannot tell it from a real one.
"""

from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from pydantic import Field

from myassistant import config
from myassistant.agents import docs_agent as da
from myassistant.state import ConfidenceTier
from myassistant.tools.observation import Observation, emit, failed


class _Scripted(BaseChatModel):
    replies: list = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = (
            self.replies[self.calls]
            if self.calls < len(self.replies)
            else AIMessage(content="done")
        )
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(*replies):
    return _Scripted(replies=list(replies))


def _call(name, tid):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "x"}, "id": tid, "type": "tool_call"}],
    )


def _tool_returning(obs):
    @tool("search_resume")
    def _t(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Fake."""
        return emit(obs, tool_call_id)

    return _t


def _run(model, tools):
    return da.build(model=model, tools=tools).invoke(
        {"messages": [HumanMessage(content="what did I do at Capital One?")]}
    )


def _good():
    return Observation(
        ok=True,
        detail="3 matches from resume_interview",
        content="[star_capitalone.docx · relevance 0.61]\nLed the EMR to Glue migration.",
        metrics={"kind": "notes", "top_score": 0.61, "threshold": 0.3},
    )


def _weak():
    return Observation(
        ok=False,
        detail="only weak matches in resume_interview (best 0.11 < 0.3)",
        content="[spark.md · relevance 0.11]\nUnrelated content.",
        metrics={"kind": "notes", "top_score": 0.11, "threshold": 0.3},
    )


def test_a_good_retrieval_earns_high():
    out = _run(
        _model(_call("search_resume", "c1"), AIMessage(content="You led...")),
        [_tool_returning(_good())],
    )
    assert out["confidence"] is ConfidenceTier.HIGH


def test_a_weak_retrieval_is_low_even_when_the_model_sounds_sure():
    """The exact danger: a confident invented answer about their own career."""
    out = _run(
        _model(_call("search_resume", "c1"), AIMessage(content="You led the migration.")),
        [_tool_returning(_weak())],
    )
    assert out["confidence"] is ConfidenceTier.LOW


def test_an_empty_collection_is_low():
    empty = failed("nothing in resume_interview matched", kind="notes", n_results=0)
    out = _run(
        _model(_call("search_resume", "c1"), AIMessage(content="You led...")),
        [_tool_returning(empty)],
    )
    assert out["confidence"] is ConfidenceTier.LOW


def test_answering_without_searching_is_low():
    """No search means no grounding - it must never read as HIGH."""
    out = _run(_model(AIMessage(content="You worked on tokenisation.")), [])
    assert out["confidence"] is ConfidenceTier.LOW


def test_the_step_cap_follows_config():
    assert da.RECURSION_LIMIT == 2 * config.MAX_TOOL_STEPS + 2


def test_the_prompt_forbids_filling_gaps_from_memory():
    """The single most important instruction this agent carries."""
    assert "Do not fill the gap from memory" in da.PROMPT
    assert "don't really cover that" in da.PROMPT


def test_the_prompt_asks_for_the_source_file():
    """An answer naming the file it came from is checkable; prose is not."""
    assert "name the file it came from" in da.PROMPT
