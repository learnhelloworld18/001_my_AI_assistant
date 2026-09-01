"""Paths, models and settings, resolved once at startup.

Two kinds of state, deliberately separate:
  ASSISTANT_HOME  fixed across launches - the knowledge base lives here
  PROJECT_ROOT    the directory you launched from - coding_agent's fence
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Global state: same no matter where you launch from ---

ASSISTANT_HOME = Path(os.environ.get("MYASSISTANT_HOME", Path.home() / ".myassistant"))
ASSISTANT_HOME.mkdir(parents=True, exist_ok=True)

# Secrets live with the global state so behaviour doesn't depend on cwd.
# The repo-local .env is a dev fallback and never overrides the global one.
load_dotenv(ASSISTANT_HOME / ".env")
load_dotenv(Path.cwd() / ".env", override=False)

CHROMA_DIR = ASSISTANT_HOME / "chroma"
MANIFEST_DB = ASSISTANT_HOME / "manifest.db"
HISTORY_FILE = ASSISTANT_HOME / "history"

# --- Per-launch state ---

# Captured once, at import. coding_agent's file/shell/git tools may not escape
# this. Calling Path.cwd() later would let the fence move if anything chdirs.
PROJECT_ROOT = Path.cwd().resolve()

# --- Models (override via env; see .env.example) ---

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "llama3.2:3b")
RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "qwen2.5:3b-instruct")
CODING_MODEL = os.environ.get("CODING_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Supervisor runs every turn, so keep it warm. Specialists unload sooner so two
# 7B models are never resident at once on 16GB.
SUPERVISOR_KEEP_ALIVE = "30m"
SPECIALIST_KEEP_ALIVE = "5m"

# Low on purpose - long tool chains feel broken in an interactive REPL.
MAX_TOOL_STEPS = 3

# --- Credentials ---

# .env.example ships placeholders, and a placeholder is non-empty - so a bare
# truthiness check would report a key as configured when it isn't.
_PLACEHOLDER_PREFIXES = ("your_", "generate_with", "fill_in")


def _is_set(value: str) -> bool:
    return bool(value) and not value.startswith(_PLACEHOLDER_PREFIXES)


LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# Tracing is optional: no keys means run without it, never crash.
LANGFUSE_ENABLED = _is_set(LANGFUSE_PUBLIC_KEY) and _is_set(LANGFUSE_SECRET_KEY)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_ENABLED = _is_set(TAVILY_API_KEY)
