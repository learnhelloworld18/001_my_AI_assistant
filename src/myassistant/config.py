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
# Was llama3.2:3b. Changed after the step-2 routing checkpoint measured it at
# 6/8 correct handoffs - and sometimes no handoff at all, the supervisor just
# answering. qwen2.5:3b-instruct got 8/8. It also matches RESEARCH_MODEL, so a
# supervisor -> research hop is no longer a model swap.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "qwen2.5:3b-instruct")
RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "qwen2.5:3b-instruct")
# Used only to WRITE code, never to call tools. Ollama reports this tag as
# tools-capable, but it emits tool calls as plain text
# ('{"name": "read_project_file", "arguments": {...}}') instead of structured
# calls, 0 times out of 4 - so an agent that reads files cannot be built on it.
# Generating code binds no tools, so the defect does not apply there, which is
# why coding_agent reads with SUPERVISOR_MODEL and writes with this one.
CODING_MODEL = os.environ.get("CODING_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # embeddings, not chat

# general_agent reuses the supervisor's model rather than loading a third one.
# That model runs every turn with a 30m keep_alive, so it is always resident -
# and general_agent exists for *fast* replies, where a 2-5s cold load would be
# most of the answer's latency.
GENERAL_MODEL = SUPERVISOR_MODEL

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

# --- RAG ---

# Below this, a retrieved chunk is treated as "my notes do not really cover
# this" rather than as an answer.
#
# Measured against a real 63-file, 1849-chunk collection rather than guessed.
# Genuine questions about the owner's own work scored 0.50-0.67; clearly
# off-topic ones (a capital city, a bread recipe) scored 0.24. Set near the
# middle of that gap, so neither a slightly weak real question nor a slightly
# lucky off-topic one flips it.
RAG_RELEVANCE_THRESHOLD = 0.37

# How many chunks to retrieve. Low on purpose: a 3B context window filled with
# eight marginal chunks answers worse than one filled with three good ones.
RAG_TOP_K = 4


def _parse_roles(raw: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse CAREER_ROLES from the environment.

    Format: "Label:fragment|fragment, Label:fragment" - the label is shown to
    the user, the fragments are matched against file paths (lower-cased, with
    separators stripped) to decide which role a document belongs to. Omit the
    fragments and the label itself is used.
    """
    roles: list[tuple[str, tuple[str, ...]]] = []
    for entry in raw.split(","):
        label, _, raw_fragments = entry.strip().partition(":")
        label = label.strip()
        if not label:
            continue
        fragments = tuple(f.strip().lower() for f in raw_fragments.split("|") if f.strip())
        roles.append((label, fragments or (label.lower().replace(" ", ""),)))
    return tuple(roles)


# The owner's career roles, most important first - read from the environment,
# never committed. This repo is public, and while employment history is not a
# secret it is personal: the same split already used for credentials, extended
# from "secret" to "personal". The repo ships the mechanism; the machine
# supplies the data.
#
# Unset means cross-role search has nothing to cover, and says so rather than
# pretending. See .env.example for the format.
CAREER_ROLES = _parse_roles(os.environ.get("CAREER_ROLES", ""))

# Chunks per role for a cross-role question. Needed because plain top-k cannot
# answer one: it ranks by similarity alone, so the best-matching document's
# near-identical chunks crowd out every other source, and whole roles go
# missing from a "walk me through my career" answer while the tier still reads
# HIGH - it certifies retrieval quality, not coverage. Retrieving per role
# turns that from a ranking problem into a coverage one.
RAG_PER_ROLE_K = 2

# --- Vision (images) ---

# Tested against a schema diagram with known contents. moondream (1.7GB)
# fabricated an entire plausible schema that was not in the image;
# granite3.2-vision:2b (1.5GB) read it correctly under one prompt phrasing and
# claimed blindness under another, taking anywhere from 16s to 964s. This one
# scored 9/9 on both a specific and a generic prompt, in 13-36s.
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")

# Unloaded quickly on purpose. It is ~3GB and runs only when an image is
# involved; keeping it resident would evict the supervisor, which runs on every
# single turn, and charge a 2-5s reload to the next text question.
VISION_KEEP_ALIVE = "2m"

# Longest edge, in pixels, before an image is sent to the model. Measured, not
# guessed: the same diagram returned 35 useless characters at 4843x5796 and at
# 2898x2421, then 2049 characters of real content at 1600px. Total pixel count
# is what breaks it, not fine detail - so downscaling works where tiling is not
# needed.
VISION_MAX_EDGE = 1600

# A reply shorter than this means the model saw nothing usable and said
# something generic about it ("The image appears to be a flowchart", 35 chars).
# It does not error, so this is the check that turns a silent failure loud -
# the same job looks_empty() does for web pages.
VISION_MIN_USEFUL_CHARS = 120

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
