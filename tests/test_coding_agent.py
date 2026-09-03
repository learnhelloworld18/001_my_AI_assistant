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
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
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


def _run(reader, tools, writer=None):
    return ca.build(
        reader=reader,
        writer=writer or _model(AIMessage(content="Here is the answer.")),
        tools=tools,
    ).invoke({"messages": [HumanMessage(content="what does safe_path do?")]})


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


def test_only_the_writer_speaks():
    """The reader's prose is discarded - two speaking nodes would show the user
    two answers, the weaker one first. This is the one-job rule, enforced."""
    read = Observation(
        ok=True, detail="read x.py", content="def f(): pass", metrics={"kind": "file"}
    )
    out = _run(
        _model(_call("c1"), AIMessage(content="READER PROSE THAT MUST NOT APPEAR")),
        [_tool_returning(read)],
        writer=_model(AIMessage(content="WRITER ANSWER")),
    )
    spoken = " ".join(str(m.content) for m in out["messages"])
    assert "WRITER ANSWER" in spoken
    assert "READER PROSE" not in spoken


def test_the_writer_is_given_what_was_read():
    """Otherwise the split would throw away the evidence it just gathered."""
    read = Observation(
        ok=True, detail="read x.py", content="UNIQUE_FILE_MARKER", metrics={"kind": "file"}
    )
    writer = _model(AIMessage(content="ok"))
    _run(_model(_call("c1"), AIMessage(content="done")), [_tool_returning(read)], writer=writer)
    assert "UNIQUE_FILE_MARKER" in str(writer.seen[0][-1].content)


def test_a_question_needing_no_file_still_gets_an_answer():
    """ "Write a SQL query for X" reads nothing - the writer answers alone."""
    writer = _model(AIMessage(content="SELECT 1;"))
    out = _run(_model(AIMessage(content="NONE")), [], writer=writer)
    assert "SELECT 1;" in " ".join(str(m.content) for m in out["messages"])
    assert out["confidence"] is ConfidenceTier.LOW  # nothing was checked


def test_evidence_is_not_double_counted():
    """The sub-agent returns the whole accumulated list, and observations has an
    add reducer - returning all of it would double every entry."""
    read = Observation(ok=True, detail="read x.py", content="x", metrics={"kind": "file"})
    out = _run(_model(_call("c1"), AIMessage(content="done")), [_tool_returning(read)])
    assert len(out["observations"]) == 1


def test_the_read_prompt_tells_it_not_to_answer():
    assert "Do not explain or answer" in ca.READ_PROMPT


def test_the_write_prompt_does_not_claim_it_can_write_files():
    """It cannot, and saying it did would be the worst kind of wrong."""
    assert "cannot create or modify files" in ca.WRITE_PROMPT
