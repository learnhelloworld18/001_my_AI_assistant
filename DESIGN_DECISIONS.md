# Important Design Considerations

Brief record of what changed from the first proposed architecture, and
why. See `REQUIREMENTS.md` for the current spec in full.

## Testing — formalized from an informal hard rule

- `CLAUDE.md` already said "test each agent node in isolation," but
  that meant "check it manually in a REPL," not a real test suite.
  Formalized into two `pytest` tiers: component tests (`tests/unit/`,
  mocked, fast, run by default and in `pre-commit`) and live dependency
  smoke tests (`tests/live/`, `@pytest.mark.live`, opt-in only) — the
  live tier exists specifically because a fully-mocked suite can pass
  while the real integration is broken, which already happened once
  (the `ddgs`/`duckduckgo-search` package-name mismatch in the GAIA
  project, invisible until tested against a real environment).

## Orchestration

- **Hand-rolled supervisor → `langgraph-supervisor`.** Originally
  planned to hand-build the routing loop for learning value. Switched
  to the prebuilt package — less plumbing to get wrong, and using it
  seriously (reading how it structures handoffs/state) is itself the
  learning value. Keeps the stack coherent: LangChain for tools,
  LangGraph for orchestration, no second framework.

## Tool sourcing

- **MCP CLI project: "reuse" → "reference only."** Originally planned
  to reuse the existing MCP CLI project's code and wrap most tools via
  `langchain-mcp-adapters`. Corrected: that project is a pattern
  reference only, not a source to copy from. Tools are built as plain
  LangChain tools (`@tool`/`Tool`) from scratch; `langchain-mcp-adapters`
  is only used if an actual external MCP server gets connected later.

## Safety — added entirely, wasn't in the original design

- **`coding_agent` safety boundary.** The first design just said "file/
  shell/git tools" with no guardrails. Added: working-directory scope,
  a confirmation gate on state-changing actions (with an explicit
  "declined" path, not a silent pass-through), and a hard denylist
  (privilege escalation, credential file access, destructive deletes).
- **Secrets & credentials — absolute hard rule.** Not addressed at all
  originally. Added: `.env` gitignored from commit #1, `.env.example`
  committed, and `gitleaks` via `pre-commit` as a technical enforcement
  layer — a written rule alone isn't enough for an absolute "never."

## RAG & memory

- **`conversation_memory` collection — added.** The original RAG design
  only had `tech_notes` + `resume_interview`; cross-session continuity
  (the REPL remembering past sessions) was discussed but never made it
  into the spec until a later pass. Session summaries (not raw
  transcripts) are stored and retrieved at next session start.
- **`resume_interview` ingestion — went from an undocumented gap to a
  concrete design.** Seeded from a real folder
  (`~/Documents/application_docs/PREP/`), which revealed real
  requirements the abstract design missed: recursive directory walk,
  a file-type allowlist, and explicit exclusion of dated `_BACKUP_*`
  files (would otherwise pollute retrieval with stale, conflicting
  content). Also clarified that not everything in a "prep" folder is
  necessarily prep content — `spark.md` there is general technical
  reference material and belongs in `tech_notes` instead.
- **Ingestion is idempotent, not append-only.** "Can I update this
  later" needed a real answer: a manifest (source path + content hash)
  plus delete-before-replace on Chroma (`.delete(where={"source": ...})`)
  before re-adding changed content — otherwise re-ingestion would leave
  stale chunks competing with current ones in retrieval.

## Tooling — went from mostly implicit to fully concrete

- Dependency management: `uv` (matches the existing MCP project).
- REPL/CLI: bare terminal + streaming `print()` as the default (Tier 0)
  — deliberately not a heavier framework, given the "just like the
  terminal, live" preference and the project's responsiveness priority.
- Web search: DuckDuckGo (`ddgs`) → **Tavily** — direct experience in
  the prior GAIA project showed DDG was flaky (DNS errors, inconsistent
  snippet quality); Tavily is LLM-optimized and more reliable.
- Linting/formatting: **`ruff` alone**, not paired with Black — two
  formatters can disagree with each other, one fast tool is simpler.
- Generated-code validation: added `validate_code.py`
  (`ruff check`/`sqlfluff`/`terraform validate`/`dbt parse`) — the
  original design had no check that generated code was even
  syntactically valid before showing it.

## Architecture diagram — created, then corrected twice

- Didn't exist as a visual artifact originally; added as a Mermaid
  diagram once the design had enough moving parts to need one.
- **First correction**: the diagram initially had a disconnected
  `/exit` flow (didn't actually connect to the session-summarization
  chain drawn elsewhere), meta-commands collapsed into one generic
  "handled directly" box hiding that each has a different real
  destination, and no visual sense that the REPL loops rather than
  running once.
- **Second correction**: the embedding step (`nomic-embed-text`) was
  entirely invisible despite being required for every RAG operation;
  the safety confirmation gate was drawn as a silent pass-through
  instead of a real decision with a decline path; `validate_code` and
  the write path were drawn as two disconnected parallel branches
  instead of the actual sequence (validate, then decide whether
  confirmation is needed). Also added color coding by component role.

## Result validation / confidence — the biggest gap, closed last

- The original design had **no validation at all** for `research_agent`,
  `docs_agent`, or `general_agent` — a deliberate early choice (dropped
  the GAIA-style evaluate/revision loop for latency reasons), but that
  choice quietly meant a wrong answer from those three agents would go
  straight to the user with nothing catching it.
- First proposed fix (a prompt-only "must call `visit_webpage`" rule for
  `research_agent`) was correctly rejected as too weak — a prompt
  instruction with no post-hoc check, same category of problem as the
  GAIA project's tool-selection compliance issues.
- Final design: **evidence-based confidence tiers**, not a raw model
  self-reported percentage. A percentage would be false precision —
  there's no calibrated probability model here, only a few real
  signals. Each agent's tier is tied to something that actually
  happened: `docs_agent` → the RAG relevance score itself,
  `research_agent` → whether `visit_webpage` actually succeeded (not
  just whether the model was told to call it), `coding_agent` → whether
  `validate_code.py` passed, `general_agent` → always honestly labeled
  ungrounded, since it has no tools to check against.

## Observability — Langfuse pulled forward from "later upgrade" to "from the start"

- Original plan: hand-roll a SQLite + timing-wrapper system first (v1),
  optionally upgrade to Langfuse later (v2) if the hand-rolled version
  proved useful. Revised: **Langfuse from the start**, once "can we see
  this visually, like a live dashboard" made clear a text-only v1 wasn't
  actually the goal — no point hand-rolling something you're going to
  replace immediately anyway.
- Attaching `CallbackHandler` to the graph captures most of the planned
  metrics (latency, token counts, tool I/O) **automatically** — the
  custom `metrics.py` timing wrapper is now mostly unnecessary. Only a
  few bespoke metrics (tool-never-chosen, RAG relevance,
  confidence-tier) still need explicit instrumentation, logged via
  Langfuse's own SDK rather than a second system.
- `/stats` keeps its originally-designed behavior (a live text summary
  printed in the terminal via the meta-command) — that requirement
  didn't go away, it just now sources its data from Langfuse's API
  instead of a hand-rolled SQLite store, so there's one system of
  record with two views (terminal + browser dashboard).
- **Named tradeoff, not a free upgrade**: self-hosted Langfuse is a
  real multi-container Docker service — the one deliberate departure
  from the "local-first, no server" pattern used everywhere else in
  this project (Chroma, SQLite). Accepted consciously for the payoff of
  a real live dashboard instead of building one.
- The RAG ingestion manifest (source path + content hash, for idempotent
  re-ingestion) still uses SQLite — that choice didn't change, it just
  became its own dedicated file (`rag/manifest.db`) instead of sharing
  a database with the now-removed observability SQLite store.

## Sub-agents — scoped down from an early broader idea

- Clarified into two separate, smaller things rather than one big
  architectural commitment: using Claude Code's own forking as a *dev
  workflow* convenience (parallelizing scaffolding), and treating Claude
  Code's supervisor→subagent dispatch as an *architectural reference*
  to study, not a v1 requirement — this project's supervisor stays
  synchronous/blocking for simplicity, revisited only if that becomes a
  real responsiveness problem.
