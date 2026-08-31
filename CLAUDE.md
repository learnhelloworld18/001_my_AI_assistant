# CLAUDE.md

Instructions for Claude Code when working in this repo. Full
architecture/context lives in `REQUIREMENTS.md` — read it before
starting significant work; this file is the condensed "how to behave
here," not a duplicate of it.

## What this project is

A personal, local-first, multi-agent AI assistant (Ollama + LangGraph +
RAG + MCP). Built for learning agentic AI patterns. **Not** graded or
benchmarked — optimize for responsiveness and "good enough," not
exhaustive correctness.

## Priorities, in order

1. **Responsiveness.** A slow "more thorough" answer is a worse outcome
   than a fast "good enough" one in this project. Default to smaller
   models, lower step caps, and streaming over maximizing accuracy.
2. **Learning value.** Orchestration/routing uses `langgraph-supervisor`
   directly (decided over hand-rolling it — less plumbing, and using it
   seriously is itself the learning value). For other mechanisms, still
   prefer hand-rolling once before reaching for a prebuilt abstraction
   that hides how it works, and note in comments/commits when something
   is deliberately hand-rolled for this reason.
3. **Reuse before rebuild — but only where that's actually the
   instruction.** Tools from the prior GAIA agent project (web_search,
   wikipedia_search, visit_webpage, etc.) should be reused/adapted
   directly, not rewritten from scratch. The existing MCP CLI project is
   different: use it as a **reference only** (patterns/conventions) —
   build new LangChain tools from scratch (`@tool`/`Tool` patterns,
   same as the GAIA `tools/` package), scoped to this project's actual
   requirements, not copied from the MCP project.

## Hard rules

- **NEVER commit or push credentials/secrets to GitHub.** This repo is
  public. `.env` must be in `.gitignore` from the first commit, before
  any code that reads a secret is written. This rule has no exceptions
  and no "just this once."
- **`coding_agent`'s file/shell/git tools require the safety boundary
  from `REQUIREMENTS.md`** (working-directory scope, confirmation gate
  on state-changing actions, hard denylist) — do not add or expand
  these tools without it in place. This agent is the only one that can
  change real state on the user's machine.
- The REPL (`main.py`) must never crash and exit on a single failed
  turn — catch model/tool/agent errors at the loop level, show a short
  message, log it, and return to the prompt.
- One job per agent/node. Never let a single agent both research and
  write the final answer — that combination reliably produces
  prose-leakage/formatting bugs (confirmed in the prior GAIA build).
- Every agentic loop (tool-calling, supervisor routing) needs an
  explicit step/iteration cap from the moment it's written, not added
  later after it misbehaves.
- No exact-match-style evaluate/revision loop. That pattern exists to
  satisfy strict grading and doesn't belong in this project — it adds
  latency for no benefit here.
- Use typed state (TypedDict or Pydantic) for any LangGraph state
  schema — don't pass around bare dicts.
- Test each agent node in isolation (a script or REPL call) before
  wiring it into the full supervisor graph.
- Keep this project's venv isolated with pinned dependencies. Don't
  assume a package is available because it's installed in some other
  project's shared environment — verify against this repo's own
  `requirements.txt`/venv.
- Keep the stack coherent: LangChain for tools, LangGraph
  (`langgraph-supervisor`) for orchestration. Don't introduce a second,
  incompatible orchestration framework.

## Models

Task-specific, not one generalist model. See `REQUIREMENTS.md` for the
current lineup and rationale (coder model for code, small/fast model
for routing, general model for research/docs, `nomic-embed-text` for
RAG). Hardware ceiling: Apple M4, 16GB RAM — avoid recommending or
defaulting to models much above 7-8B for interactive use.

## When proposing changes

- If a design decision trades learning value for convenience (e.g.
  swapping a hand-rolled piece for a prebuilt library), say so
  explicitly and let the user decide rather than silently taking the
  shortcut.
- Update `REQUIREMENTS.md` when architecture or priorities actually
  change — keep it as the living source of truth, not just an initial
  planning doc.
