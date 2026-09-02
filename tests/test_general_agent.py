"""general_agent, tested in isolation - no Ollama.

The point of this agent is what it *doesn't* claim, so most of these assert on
the honesty machinery rather than on the answer.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from myassistant import config
from myassistant.agents import general_agent as ga
from myassistant.state import ConfidenceTier


class _Scripted(BaseChatModel):
    """Returns a fixed reply and records what it was sent."""

    reply: str = "Sure - here's a draft."
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


def _run(model, text="draft me a follow-up email"):
    return ga.build(model=model).invoke({"messages": [HumanMessage(content=text)]})


def test_the_tier_is_always_ungrounded():
    assert _run(_Scripted())["confidence"] is ConfidenceTier.UNGROUNDED


def test_a_confident_technical_answer_is_still_ungrounded():
    """The tier is about what could be checked, not about how sure it sounded."""
    model = _Scripted(reply="Spark broadcasts joins under 10MB by default.")
    out = _run(model, "what's the broadcast join threshold?")
    assert out["confidence"] is ConfidenceTier.UNGROUNDED


def test_no_observations_are_produced():
    """No tools means no evidence at all.

    The key exists because reducer-backed channels are always initialised; what
    matters is that it stays empty. That is also why this agent sets its tier
    explicitly - deriving it from evidence would give LOW, and LOW would imply
    the agent tried and came up short rather than never having had tools.
    """
    assert _run(_Scripted())["observations"] == []


def test_the_system_prompt_is_prepended():
    model = _Scripted()
    _run(model)
    sent = model.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == ga.PROMPT


def test_the_conversation_is_passed_through():
    model = _Scripted()
    _run(model, "rewrite this bullet")
    assert any(getattr(m, "content", "") == "rewrite this bullet" for m in model.seen[0])


def test_the_reply_is_attributed_to_this_agent():
    """Without a name every agent's output looks identical downstream."""
    assert _run(_Scripted())["messages"][-1].name == ga.NAME


def test_the_prompt_forbids_inventing_specifics():
    """The one failure mode that matters: a confidently wrong version or default."""
    assert "do not invent specifics" in ga.PROMPT
    assert "cannot verify" in ga.PROMPT


def test_it_reuses_the_already_warm_supervisor_model():
    """A cold load would be most of the latency for the agent built to be fast."""
    assert config.GENERAL_MODEL == config.SUPERVISOR_MODEL
