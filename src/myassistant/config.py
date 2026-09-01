"""Paths, models and settings, resolved once at startup.

Two kinds of state, deliberately separate:
  ASSISTANT_HOME  fixed across launches - the knowledge base lives here
  PROJECT_ROOT    the directory you launched from - coding_agent's fence

Import this module before anything that reads an env var: load_dotenv() runs
here, and libraries like langchain-tavily read os.environ directly at init.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Global state: same no matter where you launch from ---

# Overridable via MYASSISTANT_HOME, mainly so tests can point somewhere else.
ASSISTANT_HOME = Path(os.environ.get("MYASSISTANT_HOME", Path.home() / ".myassistant"))
ASSISTANT_HOME.mkdir(parents=True, exist_ok=True)

# Secrets live with the global state so behaviour doesn't depend on cwd.
# override=False means the repo-local .env is a dev fallback only - whatever
# the global file already set wins.
load_dotenv(ASSISTANT_HOME / ".env")
load_dotenv(Path.cwd() / ".env", override=False)

CHROMA_DIR = ASSISTANT_HOME / "chroma"  # vector store (step 4)
MANIFEST_DB = ASSISTANT_HOME / "manifest.db"  # source -> hash, for idempotent re-ingest
HISTORY_FILE = ASSISTANT_HOME / "history"  # REPL input history, shared across launches

# --- Per-launch state ---

# Captured once, at import. coding_agent's file/shell/git tools may not escape
# this. Calling Path.cwd() later would let the fence move if anything chdirs -
# a shell command, a build script, a test - and the safety check would still
# report "passed" while pointing at the wrong directory.
PROJECT_ROOT = Path.cwd().resolve()

# --- Models ---
#
# Defaults live here rather than in .env because model names are not secrets:
# they are part of the project's reproducible behaviour and belong in version
# control. Still overridable per-machine via env vars (see .env.example).

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Sized for 16GB. Only the coder model is 7B - it handles the primary use case
# and is the one most hurt by a smaller model. The others are 3B so a chained
# query (supervisor -> specialist -> specialist) doesn't force an eviction.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "llama3.2:3b")
RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "qwen2.5:3b-instruct")
CODING_MODEL = os.environ.get("CODING_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # embeddings, not chat

# How long Ollama keeps a model in memory after its last use. The supervisor
# runs every turn so it stays warm; specialists unload sooner so two 7B models
# are never resident at once.
SUPERVISOR_KEEP_ALIVE = "30m"
SPECIALIST_KEEP_ALIVE = "5m"

# Max tool calls per agent turn. Low on purpose - long tool chains feel broken
# in an interactive REPL, and this project favours fast over exhaustive.
MAX_TOOL_STEPS = 3

# --- Answer critic (step 2) ---
#
# Off by default: this is the one component that adds a model call to every
# turn, against priority 1. It exists so the cost can be measured rather than
# assumed - flip it, compare critic latency in Langfuse, decide with numbers.
# Env-overridable so back-to-back timing runs need no code edit.
CRITIC_ENABLED = os.environ.get("CRITIC_ENABLED", "").lower() in ("1", "true", "yes")

# Reuses the supervisor's model on purpose. That model runs every turn with a
# 30m keep_alive so it is always resident - any other model would add Ollama's
# 2-5s cold-load to the very latency this flag exists to measure.
CRITIC_MODEL = SUPERVISOR_MODEL

# One revision, not a loop. A second pass that still fails means the evidence
# is weak, not that the wording needs another try - answer at the low tier.
CRITIC_MAX_REVISIONS = 1

# --- Self-reported confidence (calibration experiment, step 2) ---
#
# Off by default. Asks the agent for its own confidence number and logs it to
# Langfuse under its own score name - never prints it. The hard rule stands:
# what the user sees is evidence-based. This is measurement of the model, not
# a claim to anyone, and it exists to test the rule rather than assume it.
SELF_REPORT_ENABLED = os.environ.get("SELF_REPORT_ENABLED", "").lower() in ("1", "true", "yes")

# --- Credentials ---

# .env.example ships placeholders, and a placeholder is a non-empty string - so
# `if TAVILY_API_KEY:` would report an unconfigured key as configured.
_PLACEHOLDER_PREFIXES = ("your_", "generate_with", "fill_in")


def _is_set(value: str) -> bool:
    """True only for a real value, not blank and not an .env.example placeholder."""
    return bool(value) and not value.startswith(_PLACEHOLDER_PREFIXES)


LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# Tracing is optional by design: no keys (or Docker stopped) means run without
# it. Observability must never be a hard dependency of a local assistant.
LANGFUSE_ENABLED = _is_set(LANGFUSE_PUBLIC_KEY) and _is_set(LANGFUSE_SECRET_KEY)

# Web search for research_agent (step 2).
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_ENABLED = _is_set(TAVILY_API_KEY)
