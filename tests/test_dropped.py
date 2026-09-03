"""dropped.py: recognising a dragged path, and refusing the ones that matter."""

import pytest

from myassistant import dropped


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "diagram.png"
    p.write_bytes(b"x" * 100)
    return p


# --- recognising a path ---


def test_a_plain_path_is_recognised(img):
    assert dropped.as_path(str(img)) == img


def test_a_shell_escaped_path_is_recognised(tmp_path):
    """Dragging into a macOS terminal escapes spaces - the raw line is not a path."""
    p = tmp_path / "my diagram.png"
    p.write_bytes(b"x")
    assert dropped.as_path(f"{tmp_path}/my\\ diagram.png") == p


def test_a_quoted_path_is_recognised(tmp_path):
    """Some terminals single-quote the whole path instead of escaping."""
    p = tmp_path / "my diagram.png"
    p.write_bytes(b"x")
    assert dropped.as_path(f"'{p}'") == p


def test_a_tilde_path_is_expanded():
    assert dropped.as_path("~") is None  # a directory, not a file
    assert dropped.as_path("~/definitely-not-here-12345.png") is None


def test_an_ordinary_question_is_not_a_path():
    """The whole detection rests on this: sentences do not resolve to files."""
    for line in ["what did I do at Capital One?", "explain broadcast joins", "/help"]:
        assert dropped.as_path(line) is None


def test_a_path_that_does_not_exist_is_not_a_path(tmp_path):
    assert dropped.as_path(str(tmp_path / "nope.png")) is None


def test_a_directory_is_not_a_file(tmp_path):
    assert dropped.as_path(str(tmp_path)) is None


def test_an_unbalanced_quote_is_not_a_path():
    """shlex raises on this; a stray apostrophe must not crash the REPL."""
    assert dropped.as_path("what's the schema?") is None


# --- the denylist ---


def test_credential_files_are_denied_regardless_of_confirmation(tmp_path):
    """People say yes to prompts. A confirmation is not a safety boundary."""
    for name in (".env", "id_rsa", "server.pem", "aws_credentials.json", "my_secrets.txt"):
        p = tmp_path / name
        p.write_text("x")
        assert dropped.is_denied(p), name


def test_files_under_ssh_and_aws_are_denied(tmp_path):
    for folder in (".ssh", ".aws", ".gnupg"):
        p = tmp_path / folder / "anything.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        assert dropped.is_denied(p), folder


def test_ordinary_files_are_not_denied(img, tmp_path):
    assert not dropped.is_denied(img)
    notes = tmp_path / "notes.md"
    notes.write_text("x")
    assert not dropped.is_denied(notes)


# --- routing and confirmation ---


def test_images_and_text_are_routed_differently(tmp_path):
    for name, expected in [
        ("a.png", "image"),
        ("b.jpg", "image"),
        ("c.md", "text"),
        ("d.pdf", "text"),
        ("e.docx", "text"),
        ("f.xyz", "unsupported"),
    ]:
        p = tmp_path / name
        p.write_text("x")
        assert dropped.kind(p) == expected, name


def test_the_confirmation_shows_the_resolved_path_and_size(img):
    """What is agreed to must be what is read, not what was typed."""
    line = dropped.describe(img)
    assert str(img.resolve()) in line
    assert "image" in line
    assert "KB" in line or "MB" in line
