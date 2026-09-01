"""config.py: the two path constants are the ones that matter."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _config_in(cwd: Path, expr: str) -> str:
    """Import config from a given cwd and print an expression."""
    code = f"from myassistant import config; print({expr})"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "", "PYTHONPATH": str(REPO / "src"), "HOME": str(Path.home())},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_project_root_follows_launch_dir(tmp_path):
    assert _config_in(tmp_path, "config.PROJECT_ROOT") == str(tmp_path.resolve())


def test_assistant_home_ignores_launch_dir(tmp_path):
    """The knowledge base must not fragment across launch directories."""
    a = _config_in(tmp_path, "config.ASSISTANT_HOME")
    b = _config_in(REPO, "config.ASSISTANT_HOME")
    assert a == b


def test_storage_paths_sit_under_assistant_home(tmp_path):
    home = _config_in(tmp_path, "config.ASSISTANT_HOME")
    for attr in ("CHROMA_DIR", "MANIFEST_DB", "HISTORY_FILE"):
        assert _config_in(tmp_path, f"config.{attr}").startswith(home)


def test_placeholder_keys_do_not_count_as_configured():
    """.env.example placeholders are non-empty, so truthiness alone lies."""
    from myassistant import config

    assert config._is_set("tvly-realkeylooking123")
    assert not config._is_set("")
    assert not config._is_set("your_tavily_key_here")
    assert not config._is_set("fill_in_from_langfuse_ui")
    assert not config._is_set("generate_with_openssl_rand_base64_32")
