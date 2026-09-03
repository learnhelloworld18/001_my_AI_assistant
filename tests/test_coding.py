"""coding.py: the tools that touch real state, and the fence around them."""

import pytest

from myassistant import config
from myassistant.tools import coding
from myassistant.tools.safety import Verdict


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    """A throwaway PROJECT_ROOT - no test may reach the real one."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path.resolve())
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    return 42\n")
    (tmp_path / "README.md").write_text("# project\n")
    return tmp_path.resolve()


# --- reading ---


def test_a_project_file_can_be_read(project):
    obs = coding.read_file("src/main.py")
    assert obs.ok
    assert "return 42" in obs.content


def test_a_file_outside_the_root_is_refused(project):
    obs = coding.read_file("../../etc/passwd")
    assert not obs.ok
    assert "outside the working directory" in obs.detail


def test_a_credential_file_is_refused_even_inside_the_root(project):
    (project / ".env").write_text("SECRET=1")
    obs = coding.read_file(".env")
    assert not obs.ok
    assert "refusing to touch" in obs.detail
    assert "SECRET" not in obs.render()  # the contents must not leak into the refusal


def test_a_missing_file_is_an_explicit_failure(project):
    obs = coding.read_file("nope.py")
    assert not obs.ok
    assert "no such file" in obs.detail


def test_a_long_file_is_truncated_and_says_so(project):
    (project / "big.py").write_text("x" * (coding.MAX_READ_CHARS + 500))
    obs = coding.read_file("big.py")
    assert obs.ok
    assert obs.metrics["truncated"] is True
    assert "[truncated]" in obs.content


# --- listing ---


def test_files_can_be_listed_by_glob(project):
    obs = coding.list_files("src/*.py")
    assert obs.ok
    assert "src/main.py" in obs.content


def test_hidden_files_are_not_listed(project):
    (project / ".secret").write_text("x")
    assert ".secret" not in coding.list_files("*").content


def test_no_matches_is_a_failure_not_an_empty_answer(project):
    """ "No matches" alone reads to a model as a settled answer."""
    obs = coding.list_files("*.rs")
    assert not obs.ok
    assert "matches" in obs.detail


# --- writing: planned, then done ---


def test_a_write_outside_the_root_is_refused_before_anything_happens(project):
    refusal, resolved = coding.plan_write("../escape.py", "x")
    assert refusal is not None and resolved is None
    assert not (project.parent / "escape.py").exists()


def test_a_planned_write_creates_the_file(project):
    _, resolved = coding.plan_write("src/new.py", "print(1)")
    obs = coding.do_write(resolved, "print(1)")
    assert obs.ok
    assert obs.metrics["created"] is True
    assert (project / "src" / "new.py").read_text() == "print(1)"


def test_overwriting_says_so(project):
    _, resolved = coding.plan_write("README.md", "# changed")
    obs = coding.do_write(resolved, "# changed")
    assert "overwrote" in obs.detail


def test_missing_parent_directories_are_created(project):
    _, resolved = coding.plan_write("a/b/c.py", "x")
    assert coding.do_write(resolved, "x").ok


# --- shell: planned, then done ---


def test_a_read_only_command_is_planned_as_allow():
    assert coding.plan_shell("ls -la")[0] is Verdict.ALLOW


def test_a_dangerous_command_is_planned_as_deny():
    assert coding.plan_shell("sudo rm -rf /")[0] is Verdict.DENY


def test_a_command_runs_from_the_project_root(project):
    """A relative path in a command must mean what it means to safe_path."""
    obs = coding.do_shell("ls")
    assert obs.ok
    assert "README.md" in obs.content


def test_a_failing_command_reports_its_output_not_an_exception(project):
    """The model needs the error text to decide what to do next."""
    obs = coding.do_shell("ls definitely-not-here")
    assert not obs.ok
    assert obs.metrics["exit_code"] != 0


def test_a_hanging_command_times_out(project, monkeypatch):
    monkeypatch.setattr(coding, "TIMEOUT_S", 1)
    obs = coding.do_shell("sleep 5")
    assert not obs.ok
    assert "timed out" in obs.detail


def test_the_bound_tools_are_read_only_for_now():
    """Writing needs a confirmation the agent loop cannot pause for."""
    names = {t.name for t in coding.build_tools()}
    assert names == {"list_project_files", "read_project_file"}
