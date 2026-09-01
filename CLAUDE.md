# CLAUDE.md

Instructions for Claude Code when working in this repo. Full
architecture/context lives in `project_docs/PROJECT_REQUIREMENTS.md` — read it before
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
  from `project_docs/PROJECT_REQUIREMENTS.md`** (working-directory scope, confirmation gate
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
- No exact-match-style answer-format grading. That machinery exists to
  satisfy strict benchmarks and doesn't belong here.
- **The agent loop is `reason → act → observe → reason`. There is no
  fourth "evaluate" step inside the model** — evaluation *is* the
  reasoning step running again with the observation in context. Never
  prompt for an explicit evaluation phase; it produces text that looks
  like scrutiny while adding no information the next reasoning step
  didn't already have.
- **Therefore the observation text is the engineering surface.** The
  model's self-assessment is bounded by what's legible in it — a tool
  that fails quietly and returns something plausible leaves nothing to
  notice, and the step cap won't save you, because an agent that
  believes it succeeded stops looping. **Every tool returns an
  `Observation` (`tools/observation.py`), never a bare string.** Failure
  is explicit, success carries its numbers, and a fallback must never
  masquerade as content. A tool that can fail silently is a bug even if
  it never raises.
- **The evidence gate is in the graph, not the model.** Deterministic,
  reading `Observation.ok`/`metrics` — did `visit_webpage` return real
  content, is the RAG score above threshold, did `validate_code.py`
  pass. It's what catches the failures the model can't see, so it must
  never depend on the model having noticed anything. Exhausting the step
  cap is not an error: answer anyway at the low confidence tier.
- **The model critic is opt-in and off by default** (`CRITIC_ENABLED`
  in `config.py`). It exists to *measure* what a per-turn critique
  costs rather than assume it — the one place this project spends a
  model call against priority 1, and only with the numbers to justify
  it. Capped at `CRITIC_MAX_REVISIONS` (1). It must be given the tool
  observations, not just the answer text: a model grading prose with no
  ground truth is the same uncalibrated thing the confidence rule below
  rejects. And it shares the loop's blind spot — reading the same
  observation text, it has no better access to whether a tool really
  worked. Its only real edge is a fresh context without the agent's
  committed narrative.
- **Confidence tags shown to the user are evidence-based, never a raw
  model self-report or an invented percentage.** The rule is about what
  is displayed. A self-reported number may be *logged* to Langfuse under
  `SELF_REPORT_SCORE`, behind `SELF_REPORT_ENABLED` (default off) — that
  is measurement of the model, not a claim to anyone, and it exists to
  test this rule rather than assume it. Never merge that score with the
  evidence-based one, and never print it. Each agent's confidence signal
  is tied
  to something that actually happened (RAG relevance score, whether
  `visit_webpage` succeeded, whether `validate_code.py` passed) — see
  `project_docs/PROJECT_REQUIREMENTS.md`'s Confidence & validation section. Asking a model
  to grade its own certainty as "X% confident" is not acceptable here;
  it's uncalibrated and reads as more rigorous than it is.
- Use typed state (TypedDict or Pydantic) for any LangGraph state
  schema — don't pass around bare dicts.
- Test each agent node in isolation before wiring it into the full
  supervisor graph — as a real `pytest` component test (`tests/unit/`,
  mocked dependencies), not just a manual REPL check. Live-service tests
  (`tests/live/`, `@pytest.mark.live`) are opt-in only, never part of
  the default test run or the `pre-commit` hook.
- Keep this project's venv isolated with pinned dependencies. Don't
  assume a package is available because it's installed in some other
  project's shared environment — verify against this repo's own
  `requirements.txt`/venv.
- Keep the stack coherent: LangChain for tools, LangGraph
  (`langgraph-supervisor`) for orchestration. Don't introduce a second,
  incompatible orchestration framework.

## Models

Task-specific, not one generalist model. See `project_docs/PROJECT_REQUIREMENTS.md` for the
current lineup and rationale (coder model for code, small/fast model
for routing, general model for research/docs, `nomic-embed-text` for
RAG). Hardware ceiling: Apple M4, 16GB RAM — avoid recommending or
defaulting to models much above 7-8B for interactive use.

## Explaining things

**If the answer is a flow — something moving through steps — draw the
flow first, then explain the parts.** Prose describing a pipeline makes
the reader rebuild the diagram in their head. Show the shape, then say
why each box is there. This is a learning project; the picture is the
teaching.

Applies to: "what is X", "who/what does X", "how does X work", "what
happens when I…", anything with a sequence, a branch, or a handoff. Use
a plain ASCII flow with `↓` and `→`, real function and type names from
the code, and a branch at the end when the outcome differs. Keep it
narrow enough to read in a terminal.

The example that prompted this rule — "what is the critic?":

```
you ask something
   ↓
research_agent  → calls web_search, visit_webpage
                → produces an answer + a list of Observations
   ↓
critic          → gets three things:
                    1. your question
                    2. the agent's draft answer
                    3. render_evidence(observations)  ← the actual tool output
                → returns Verdict(supported=True/False, issue="...")
   ↓
if supported  → you see the answer
if not        → one revision (CRITIC_MAX_REVISIONS), then answer anyway at LOW
```

What makes it work: concrete names over abstractions (`render_evidence`,
not "the evidence"), the inputs enumerated where the reader would ask
"but what does it actually see", and both branches shown so the failure
path isn't a mystery. Follow it with the short *why* — why it's a
separate node, why binary and not a score, what its limits are — but
only after the shape is on the page.

## When proposing changes

- If a design decision trades learning value for convenience (e.g.
  swapping a hand-rolled piece for a prebuilt library), say so
  explicitly and let the user decide rather than silently taking the
  shortcut.
- Update `project_docs/PROJECT_REQUIREMENTS.md` when architecture or priorities actually
  change — keep it as the living source of truth, not just an initial
  planning doc.
- **Show proposed changes before making them.** Before creating or
  editing any file (code, config, or otherwise), describe what's about
  to change — new file contents or a summary of the edit — and wait for
  confirmation before writing it. Applies from the repo-scaffolding step
  onward, not just docs.
- **Commit messages must be relevant, not generic.** Every commit/push
  includes a message describing what actually changed and why — not a
  placeholder like "update files."
- **Comment new code as you write it.** This is a learning project — every
  new module, class, and non-obvious block gets a comment explaining what
  it does and why it's built that way. Don't leave code bare and add
  comments in a later pass.
- **But keep them brief.** Short and simple; expand only when asked for
  detail. A comment states what the line is for and any non-obvious
  gotcha — not the full rationale. Rationale that needs paragraphs belongs
  in `project_docs/DESIGN_DECISIONS.md`, not inline.
