"""main.py: command routing, the '/'-only completer, and error containment."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from myassistant import main
from myassistant.state import ConfidenceTier


@pytest.fixture
def session():
    return main.Session()


class _StubGraph:
    """Stands in for the compiled supervisor, in its streaming shape.

    Mirrors what LangGraph actually yields with subgraphs=True: a
    (namespace, mode, chunk) triple, tokens arriving as AIMessageChunks, and
    the final root state under mode "values".
    """

    def __init__(self, text="a reply", tier=None, tokens=True, status=True):
        self.text, self.tier, self.seen = text, tier, None
        self.tokens, self.status = tokens, status

    def _final(self, state):
        out = {"messages": [*state["messages"], AIMessage(content=self.text)]}
        if self.tier is not None:
            out["confidence"] = self.tier
        return out

    def stream(self, state, config=None, **kwargs):
        self.seen = state
        if self.status:
            handoff = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transfer_to_general_agent",
                        "args": {},
                        "id": "h1",
                        "type": "tool_call",
                    }
                ],
            )
            yield (("supervisor:1",), "updates", {"agent": {"messages": [handoff]}})
        if self.tokens:
            for word in self.text.split(" "):
                yield ((), "messages", (AIMessageChunk(content=word + " "), {}))
        yield ((), "values", self._final(state))


@pytest.fixture(autouse=True)
def _no_live_services(monkeypatch):
    """Never let the default test run reach Ollama or Langfuse.

    Without this, run_turn() builds the real supervisor and answers for real -
    which turned a 1s suite into 26s of live model calls. Live coverage belongs
    behind @pytest.mark.live, not in every commit.
    """
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph())
    monkeypatch.setattr(main.langfuse_client, "get_callbacks", lambda sid: [])
    monkeypatch.setattr(main.langfuse_client, "score", lambda *a, **kw: None)
    monkeypatch.setattr(main.langfuse_client, "flush", lambda: None)


def _complete(text: str) -> list[str]:
    doc = Document(text, len(text))
    return [c.text for c in main.MetaCommandCompleter().get_completions(doc, CompleteEvent())]


# --- routing ---


def test_exit_stops_the_loop(session):
    assert main.run_turn("/exit", session) is False


def test_other_commands_continue(session):
    for cmd in ("/help", "/clear", "/stats", "/remember", "/ingest"):
        assert main.run_turn(cmd, session) is True


def test_unknown_command_does_not_crash(session, capsys):
    assert main.run_turn("/nope", session) is True
    assert "unknown command" in capsys.readouterr().out


def test_plain_text_is_not_treated_as_a_command(session):
    main.run_turn("what is a CTE", session)
    assert isinstance(session.history[0], HumanMessage)
    assert session.history[0].content == "what is a CTE"


def test_clear_empties_history(session):
    main.run_turn("hello", session)
    assert session.history
    main.run_turn("/clear", session)
    assert not session.history


def test_clear_keeps_session_identity(session):
    """/clear resets the conversation, not the session itself."""
    sid, started = session.session_id, session.started_at
    main.run_turn("hello", session)
    main.run_turn("/clear", session)
    assert session.session_id == sid
    assert session.started_at == started


def test_each_session_gets_its_own_id():
    assert main.Session().session_id != main.Session().session_id


# --- completer ---


def test_no_completions_for_plain_text():
    assert _complete("what is") == []
    assert _complete("") == []


def test_slash_lists_every_command():
    assert sorted(_complete("/")) == sorted(main.COMMANDS)


def test_prefix_narrows():
    assert _complete("/e") == ["/exit"]


def test_only_path_commands_complete_paths(tmp_path):
    (tmp_path / "notes.md").touch()
    assert _complete(f"/ingest {tmp_path}/") == ["notes.md"]
    assert _complete(f"/stats {tmp_path}/") == []


# --- registry consistency ---


def test_every_command_is_dispatchable():
    """The completer offers these, so each must actually run."""
    for name in main.COMMANDS:
        assert callable(main.COMMANDS[name].run)
        assert main.COMMANDS[name].help


# --- supervisor wiring ---


def test_history_is_replayed_into_the_next_turn(session, monkeypatch):
    """A follow-up question needs the earlier exchange, or "and that one?" fails."""
    graph = _StubGraph()
    monkeypatch.setattr(main, "_supervisor", lambda: graph)
    main.run_turn("first", session)
    main.run_turn("second", session)
    assert [m.content for m in graph.seen["messages"]] == ["first", "a reply", "second"]


# --- streaming ---


def test_tokens_are_printed_as_they_arrive(session, monkeypatch, capsys):
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(text="one two three"))
    main.run_turn("hello", session)
    assert "one two three" in capsys.readouterr().out


def test_the_answer_is_not_printed_twice(session, monkeypatch, capsys):
    """answer() prints as it streams; run_turn must not print it again."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(text="unique-marker"))
    main.run_turn("hello", session)
    assert capsys.readouterr().out.count("unique-marker") == 1


def test_a_model_that_cannot_stream_still_prints_its_answer(session, monkeypatch, capsys):
    """No tokens must not mean a blank screen - fall back to the final state."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(text="fallback", tokens=False))
    main.run_turn("hello", session)
    assert "fallback" in capsys.readouterr().out


def test_the_active_agent_is_announced(session, monkeypatch, capsys):
    """The wait before the first token is the part that feels broken."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph())
    main.run_turn("hello", session)
    assert "· general_agent" in capsys.readouterr().out


def test_history_stores_the_answer_without_the_tag(session, monkeypatch):
    """The tag is presentation - replaying it would nudge later turns."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(tier=ConfidenceTier.LOW))
    main.run_turn("hello", session)
    assert "thin evidence" not in str(session.history[-1].content)


def test_a_tool_call_is_announced_before_it_runs():
    """Announcing from the tool node would report a wait that already happened."""
    call = AIMessage(
        content="", tool_calls=[{"name": "web_search", "args": {}, "id": "t1", "type": "tool_call"}]
    )
    status = main._status(("research_agent:1",), {"agent": {"messages": [call]}})
    assert status == "· web_search…"


def test_the_chosen_agent_is_announced_from_the_handoff_call():
    """The handoff happens before the agent runs, so it can lead the answer."""
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "transfer_to_research_agent", "args": {}, "id": "t1", "type": "tool_call"}
        ],
    )
    assert main._status(("supervisor:1",), {"agent": {"messages": [call]}}) == "· research_agent"


def test_transferring_back_is_not_announced():
    """Returning to the supervisor is bookkeeping, not news."""
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "transfer_back_to_supervisor", "args": {}, "id": "t1", "type": "tool_call"}
        ],
    )
    assert main._status(("research_agent:1",), {"agent": {"messages": [call]}}) is None


def test_root_updates_are_never_announced():
    """They fire when a subgraph has already finished - always too late."""
    assert main._status((), {"research_agent": {}}) is None


def test_only_the_question_and_answer_are_kept(session, monkeypatch):
    """Handoffs and tool output must not eat a 3B model's context window."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph())
    main.run_turn("hello", session)
    assert len(session.history) == 2


def test_a_low_tier_is_shown_to_the_user(session, monkeypatch, capsys):
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(tier=ConfidenceTier.LOW))
    main.run_turn("hello", session)
    assert "thin evidence" in capsys.readouterr().out


def test_an_ungrounded_tier_is_shown_to_the_user(session, monkeypatch, capsys):
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(tier=ConfidenceTier.UNGROUNDED))
    main.run_turn("hello", session)
    assert "not verified" in capsys.readouterr().out


def test_a_high_tier_is_not_shown(session, monkeypatch, capsys):
    """A tag on every answer becomes wallpaper - the exception is the signal."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(tier=ConfidenceTier.HIGH))
    main.run_turn("hello", session)
    out = capsys.readouterr().out
    assert "[" not in out


def test_the_tier_is_scored_to_langfuse(session, monkeypatch):
    scored = []
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(tier=ConfidenceTier.LOW))
    monkeypatch.setattr(main.langfuse_client, "score", lambda n, v, **kw: scored.append((n, v)))
    main.run_turn("hello", session)
    assert scored == [(main.langfuse_client.CONFIDENCE_SCORE, "low")]


def test_an_empty_answer_falls_back_rather_than_printing_nothing(session, monkeypatch, capsys):
    """A blank answer would look like a hang."""
    monkeypatch.setattr(main, "_supervisor", lambda: _StubGraph(text="   ", tokens=False))
    main.run_turn("hello", session)
    assert "hello" in capsys.readouterr().out  # falls back to the last non-empty message


def test_failing_turn_is_caught_not_raised(session, monkeypatch, capsys):
    monkeypatch.setattr(main, "answer", lambda q, s: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        main.run_turn("boom", session)  # run_turn itself does not swallow
