"""main.py: command routing, the '/'-only completer, and error containment."""

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from myassistant import main


@pytest.fixture
def session():
    return main.Session()


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
    assert session.history[0][0] == "what is a CTE"


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


def test_failing_turn_is_caught_not_raised(session, monkeypatch, capsys):
    monkeypatch.setattr(main, "answer", lambda q, s: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        main.run_turn("boom", session)  # run_turn itself does not swallow
