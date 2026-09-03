"""coding_agent, tested in isolation - no Ollama, no filesystem."""

from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from pydantic import Field

from myassistant import config
from myassistant.agents import coding_agent as ca
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


def _call(tid):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_project_file", "args": {"path": "x.py"}, "id": tid, "type": "tool_call"}
        ],
    )


def _tool_returning(obs):
    @tool("read_project_file")
    def _t(path: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Fake."""
        return emit(obs, tool_call_id)

    return _t


def _run(model, tools):
    return ca.build(model=model, tools=tools).invoke(
        {"messages": [HumanMessage(content="what does safe_path do?")]}
    )


def test_reading_a_real_file_earns_high():
    read = Observation(
        ok=True,
        detail="read safety.py",
        content="def safe_path(...)",
        metrics={"kind": "file", "chars": 900},
    )
    out = _run(
        _model(_call("c1"), AIMessage(content="It resolves and checks.")), [_tool_returning(read)]
    )
    assert out["confidence"] is ConfidenceTier.HIGH


def test_a_refused_path_is_low_even_though_the_model_answered():
    """The fence holding must not read as a grounded answer."""
    refused = failed("/etc/passwd is outside the working directory", kind="file")
    out = _run(
        _model(_call("c1"), AIMessage(content="It checks paths.")), [_tool_returning(refused)]
    )
    assert out["confidence"] is ConfidenceTier.LOW


def test_answering_without_reading_anything_is_low():
    """The exact failure seen live: the model describing code it never opened."""
    out = _run(_model(AIMessage(content="safe_path probably validates paths.")), [])
    assert out["confidence"] is ConfidenceTier.LOW


def test_the_step_cap_follows_config():
    assert ca.RECURSION_LIMIT == 2 * config.MAX_TOOL_STEPS + 2


def test_the_prompt_says_to_look_before_answering():
    assert "Look before you answer" in ca.PROMPT


def test_the_prompt_forbids_narrating_the_handoff():
    """Seen live: "it seems a request was transferred to another agent"."""
    assert "Do not narrate" in ca.PROMPT


def test_the_prompt_does_not_claim_it_can_write_files():
    """It cannot yet, and saying it did would be the worst kind of wrong."""
    assert "cannot create or modify files" in ca.PROMPT
