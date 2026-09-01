# Personal Local AI Assistant — Requirements

## Purpose

A personal, local-first multi-agent assistant for learning agentic AI
patterns (agents, MCP, RAG, LangChain/LangGraph) while being genuinely
useful day to day. **For personal learning — prioritize responsiveness
over accuracy.** This is not a graded/benchmarked system.

## User context

- Senior Data Engineer. Works heavily in Python, SQL, PySpark, dbt,
  Terraform.
- Hardware: Apple M4, 16GB RAM (realistic ceiling: ~7-8B models
  comfortably, 14B Q4 if patient, avoid 27B+ for interactive use).
- Already has Ollama installed, `qwen2:7b` pulled, an existing MCP CLI
  project, and hands-on LangGraph experience from a prior agent build.

## Core use cases

- Coding help — Python, SQL, PySpark, dbt, Terraform.
- Document editing / formatting.
- Resume point suggestions / editing.
- Interview prep (e.g. mock Q&A).
- Writing emails.
- Researching online.
- Reading/gathering and later querying accumulated tech knowledge.

## Interaction model

- Entry point: `main.py`, run from the terminal — **installed globally**
  (`uv tool install .`, backed by its own isolated venv), launchable
  from any directory on the Mac, not just this repo. See Global
  installation & scope below for what that means for state.
- Conversational REPL, similar in feel to Claude Code — fast, streaming
  responses, iterative back-and-forth.
- Responses should feel near-instant for simple queries (small models,
  warm `keep_alive`, streaming output, capped context window).
- **Meta-commands**: not everything routes through an agent. Support
  slash-style commands handled directly by `main.py` — e.g. `/help`,
  `/exit`, `/stats` (recent observability summary), `/ingest <path>`
  (trigger RAG ingestion), `/clear` (reset session state), `/remember
  [text]` (save to long-term memory now, on request — see Explicit
  memory saves below). These bypass the supervisor entirely for speed
  and predictability.
- **Autocomplete for meta-commands**: typing `/` shows a dropdown of
  available commands (narrows as you type — `/i` → `/ingest`), and the
  `<path>` argument to `/ingest` gets filesystem-path completion, same
  idea as shell tab-completion. Brings in `prompt_toolkit` for this
  specifically — the Tier 1 upgrade from Tooling stack, pulled forward
  from "later if it bugs you" to now. **Only the input side changes** —
  normal conversational input (not starting with `/`) gets no
  completion popup, staying fast/uncluttered for the common case, and
  output stays plain streaming `print()` (not adopting `rich` or
  anything heavier — that's still deferred).
- **Error handling**: any model call, tool call, or agent failure must
  be caught at the REPL loop level, shown as a short user-facing
  message, and logged (see Observability) — the REPL must never crash
  and exit on a single failed turn. The loop always returns to the
  prompt.

## Global installation & scope — two kinds of state, not one

Launching from any directory (like Claude Code) means being explicit
about what stays constant across launches and what changes per launch —
conflating the two would either fragment your knowledge base across
every directory you ever run the tool from, or let `coding_agent` touch
files outside wherever you actually meant to work.

- **Global, persistent** (lives in a fixed location, e.g.
  `~/.myassistant/`, **not** `cwd`-relative): the Chroma vector store
  (`tech_notes`, `resume_interview`, `conversation_memory`),
  `rag/manifest.db`, `.env` secrets, Langfuse config. This is the whole
  point of a personal knowledge base — it has to persist regardless of
  which directory you happened to launch from.
- **Per-launch, `cwd`-scoped**: `coding_agent`'s working-directory
  boundary (the existing Safety boundaries requirement) is
  `Path.cwd()` **at launch time** — this is the actual mechanism for
  "access the contents of that repo." Not a new capability, just making
  explicit that "project root" in the Safety boundaries section means
  wherever you launched from, not the assistant's own install location.
- Other agents (`research_agent`, `docs_agent`, `general_agent`) that
  need repo content route through `coding_agent`'s already-scoped tools
  via supervisor chaining, rather than each agent growing its own
  separate file-access logic — one safety boundary, not several.
- **Installation**: `uv tool install .` registers a real executable on
  `$PATH`, backed by its own isolated venv — needs a `[project.scripts]`
  entry in `pyproject.toml` and a proper `main()` function (currently
  just `uv init`'s placeholder stub).

## Architecture: multi-agent, supervisor pattern

Not a single hard router — a **supervisor** that can call multiple
specialist agents within one query and chain their outputs (e.g.
research a topic, then hand the findings to the coding agent).

```
supervisor (fast/small model, decides which agent acts next)
   ├─→ coding_agent    (coder-tuned model + file/shell/git tools)
   ├─→ research_agent  (general model + web_search/wikipedia_search tools)
   ├─→ docs_agent      (general model + RAG tool over personal notes/resume)
   └─→ general_agent   (small/fast model, no tools — quick chat, email drafts)
```

- Use LangGraph's prebuilt `langgraph-supervisor` package directly for
  orchestration/routing, rather than hand-rolling it — less plumbing to
  get right, and using it seriously (reading how it structures handoffs
  and state) is itself the learning value here, not rebuilding it from
  scratch.
- Each agent has exactly one job — do not combine research and
  answer-writing (or any two responsibilities) in a single agent/prompt.
- **Coherent stack, deliberately**: tools are built with LangChain,
  orchestration with LangGraph (`langgraph-supervisor` is built on
  LangGraph). Don't introduce a second, incompatible orchestration
  framework — the whole point of this choice is that the tool layer and
  the orchestration layer speak the same abstractions.

See `ARCHITECTURE.md` for the full request-flow diagram.

## Models (task-specific, not one generalist)

| Role | Model | Notes |
|---|---|---|
| Coding | `qwen2.5-coder:7b-instruct-q4_K_M` | Python/SQL/PySpark/dbt/Terraform |
| Research / docs | `qwen2.5:3b-instruct` or `qwen2.5:7b` | general reasoning |
| Supervisor / quick chat | `llama3.2:1b` or `:3b` | fast routing decisions, low-stakes drafts |
| Embeddings (RAG) | `nomic-embed-text` | not a chat model |

## RAG — personal knowledge base

- Ingest accumulated tech notes/articles → chunk → embed
  (`nomic-embed-text`) → store in Chroma (local, no server needed).
- Query at runtime: embed the question, retrieve top-k chunks, hand to a
  chat model as context.
- Also usable specifically for resume/interview-prep content.
- Distinct from tool-based retrieval (web_search/wikipedia_search) —
  RAG is always-on retrieval before generation; tool-based retrieval is
  an on-demand agent decision. Understand and use both patterns.

**Collections** (separate, not one blob — each has its own ingestion
path and metadata):

| Collection | Contents | Used by |
|---|---|---|
| `tech_notes` | Articles/notes you read, chunked | `docs_agent` |
| `resume_interview` | Resume bullets, past interview answers, job descriptions | `docs_agent` |
| ↳ seed source | `~/Documents/application_docs/PREP/` — company-specific interview prep + `STAR/` resume-points docs + `01_extra notes/<company>/`. **Not** `spark.md` in that same folder — that's general Apache Spark reference material, not interview-specific, and belongs in `tech_notes` via its own ingest call instead |
| `conversation_memory` | Per-session summaries (not raw transcripts) | supervisor / any agent, for cross-session continuity |

**Conversation memory, specifically**:
- At the end of a REPL session (or on `/exit`), summarize that
  session with a cheap fast model (e.g. `llama3.2:1b`) and store the
  summary in `conversation_memory`.
- At the start of a new session, retrieve relevant past summaries so
  the assistant has continuity across restarts — the REPL is **not**
  a blank slate every launch.
- This is a deliberately different pattern from `tech_notes`: store a
  distilled summary, not the raw conversation, to keep retrieval signal
  clean.

**Ingestion mechanics (`/ingest <path> [--collection <name>]`)**:
- `--collection` defaults to `tech_notes` if omitted; pass
  `--collection resume_interview` for the PREP folder.
- Walks subdirectories recursively (the PREP folder has nested
  structure: `STAR/`, `01_extra notes/<company>/`) — not top-level
  files only.
- File-type allowlist: `.md`, `.docx`, `.pdf`, `.txt` only. Everything
  else (`.DS_Store`, `images/*.png`, etc.) is silently skipped, not an
  error.
- **Backup-file exclusion**: skip any file matching a `*_BACKUP_*`
  naming pattern (e.g. `resume_points_STAR_michelin_BACKUP_2026-08-09.docx`)
  — ingesting dated backups alongside current versions would put stale,
  conflicting resume points into retrieval right next to the current
  ones.

**Explicit memory saves (`/remember [text]`)**:
- On-demand counterpart to the automatic end-of-session summarization
  above — the user shouldn't have to wait for `/exit` to persist
  something worth keeping.
- `/remember` with no arguments: summarize the conversation so far
  (same cheap fast model as end-of-session) and store it.
- `/remember <text>`: store the given text verbatim, no
  summarization step — the user has already distilled it themselves.
- Both forms always target `conversation_memory` only — never
  `tech_notes` or `resume_interview`, which have their own deliberate
  ingestion path via `/ingest`.

## Confidence & validation (evidence-based, not model self-report)

No agent gets a full correctness-grading loop (deliberately dropped —
see Speed-first design decisions), but every agent that has a *real,
measurable signal* available surfaces it. **Not** a raw percentage —
there is no calibrated probability model here, only a few concrete
signals, and presenting those as "82% confident" would be false
precision, not honesty. Tiers instead, each tied to something that
actually happened, not the model grading itself:

| Agent | Confidence signal | High tier | Low tier |
|---|---|---|---|
| `docs_agent` | RAG relevance score (the same number that gates whether a chunk is used at all) | above threshold — labeled grounded in `tech_notes`/`resume_interview` | below threshold — must say "my notes don't cover this well," not present a weak match as fact |
| `research_agent` | did `visit_webpage` actually succeed on a real page, vs. only search snippets | visited and extracted specific content | only got snippets, or a visit failed |
| `coding_agent` | did `validate_code.py` pass | validation passed cleanly | validation failed, or wasn't run |
| `general_agent` | none — no tools, no grounding is possible | — | always labeled "general knowledge, not verified against your data" |

- This is a genuine correction from an earlier, weaker version of this
  idea (a prompt-only instruction for `research_agent` with no
  post-hoc check) — the signal now reflects what the agent actually
  did, not just what it was told to do.
- Surfaced as a short tag on the response only when it's actually
  informative — low-confidence answers get a visible note; the
  `general_agent` tag is always shown since it's always true. No extra
  LLM call for any of this — every signal already exists as a
  byproduct of the tool call that already happened.

**Updating ingested content is doable and logically correct — with one
real requirement**: naive re-ingestion (just adding new embeddings) is
*not* enough on its own, because it would leave old chunks from a
since-edited file sitting in Chroma alongside the new ones, and
retrieval would return both — stale and current versions of the same
resume bullet competing in search results. The actual mechanism:
- Track a manifest — `(source path, content hash, collection)` — in a
  small local SQLite file (`rag/manifest.db`). This is now its own
  dedicated file, not shared with observability — Langfuse replaced the
  observability SQLite store, but this manifest still fits SQLite well
  (simple structured local record, no server needed).
- On each `/ingest` run: unchanged files are skipped (hash matches, no
  work); changed or new files are re-chunked/re-embedded, and Chroma's
  metadata-filtered delete (`.delete(where={"source": path})`) removes
  that file's *old* chunks **before** the new ones are added — never
  append-only.
- This means re-running `/ingest ~/Documents/application_docs/PREP
  --collection resume_interview` any time those docs change is the
  correct way to keep the collection current, not a special "update"
  command — ingestion is idempotent and safe to re-run.

## Tool supply — LangChain tools, built from scratch

- Use the existing MCP CLI project as a **reference only** (patterns,
  conventions) — do not reuse its code directly.
- **"Building from scratch" means writing new LangChain tools**
  (`@tool` decorator / `Tool` class — same pattern as the GAIA
  project's `tools/` package), scoped to this project's actual
  requirements. This is the default construction method for every tool.
- `langchain-mcp-adapters` is used only if/when this project connects
  to an actual external MCP server later — it is not the default way
  tools get built here.
- Filesystem, git, shell execution tools for `coding_agent` — see
  **Safety boundaries** below, non-negotiable for these specifically.
  Also gets a **`validate_code` tool** (see Tooling stack) that lints
  generated Python/SQL/Terraform/dbt before it's shown or written —
  code correctness is a core use case, not an afterthought.
- Web search via **Tavily** (LLM-optimized results, meaningfully more
  reliable than DuckDuckGo based on direct experience in the prior GAIA
  build — repeated DNS errors and inconsistent snippet quality there).
  `page-visiting` tool (`visit_webpage`) reused as-is from that project.
  Needs `TAVILY_API_KEY` in `.env`.
- RAG tools for `docs_agent` — one tool per collection (`search_notes`,
  `search_resume`), so the model's tool choice signals which collection
  to search.

## Safety boundaries — `coding_agent` (hard requirement, not optional)

`coding_agent` is the only agent that can change real state on your
machine (files, git history, shell commands) — it needs guardrails from
first implementation, not bolted on after something goes wrong.

- **Working-directory scope**: file/shell/git tools operate only within
  an explicitly configured project root — concretely, `Path.cwd()` at
  launch time (see Global installation & scope), not the assistant's
  own install location. No arbitrary absolute paths, no traversal
  outside that root.
- **Confirmation gate**: any state-changing action (file write/delete,
  `git commit`/`push`/`reset`, any shell command that isn't read-only)
  must be shown to the user and explicitly confirmed in the REPL before
  executing. Read-only operations (file read, `git status`/`diff`/`log`)
  do not require confirmation — keep friction only where it matters, to
  preserve responsiveness.
- **Hard denylist**: some actions are never executable, confirmation or
  not — destructive recursive deletes, `sudo`/privilege escalation, and
  reading/exfiltrating credential or secret files (see Secrets &
  Credentials below).

## Secrets & Credentials — hard rule: NEVER pushed to GitHub

This repo is public. This rule is absolute, not a best-effort guideline.

- `.env` (and any file holding API keys/tokens) must be in `.gitignore`
  from the **very first commit** — before any code that reads a secret
  is written.
- Provide `.env.example` with placeholder values only, committed, so the
  real `.env` structure is documented without exposing anything.
- `coding_agent`'s denylist (above) explicitly blocks reading `.env` or
  any credential file, even if asked to "look at the config."
- Before every commit/push (agent-initiated or manual), verify no
  secret-looking content is staged — same discipline as the GAIA
  project's git workflow.
- **Technical enforcement, not just discipline**: `gitleaks` wired in
  via the `pre-commit` framework blocks a commit containing
  secret-looking content automatically — a written rule alone isn't
  enough of a safeguard for an absolute "NEVER."

## Tooling stack

Concrete libraries/tools per concern — avoid leaving any of these
implicit.

| Concern | Tool | Notes |
|---|---|---|
| Dependency management | `uv` | matches the existing MCP CLI project's convention (`pyproject.toml` + lockfile) |
| REPL/CLI | `prompt_toolkit` for input (history + meta-command/path autocomplete), plain streaming `print()` for output | Tier 1 input, Tier 0 output — see Interaction model. `rich`-style formatted output (Tier 2) still deferred |
| LLM serving | Ollama | already set up; `langchain-ollama` (`ChatOllama`) is the integration point |
| Orchestration | LangGraph + `langgraph-supervisor` | |
| Tools | LangChain (`@tool`/`Tool`) | see Tool supply |
| Vector DB | Chroma | |
| Document parsing | `pypdf` (PDF), `python-docx` (DOCX), plain read (MD) | lightweight, per-format — not the heavier `unstructured` package unless these prove insufficient |
| Web search | Tavily (`TAVILY_API_KEY`) | see Tool supply for rationale |
| Observability | Langfuse (self-hosted, Docker) | `CallbackHandler` on the graph; `/stats` reads back via Langfuse's API — see Observability section |
| Testing | `pytest` — component tests (mocked, default) + live dependency smoke tests (`@pytest.mark.live`, opt-in via `pytest -m live`) | see Testing section |
| Linting + formatting | `ruff` (both — not paired with Black, to avoid two tools disagreeing on style) | |
| Type checking | `mypy` via `pre-commit` | pairs with the typed-state hard rule — static code analysis, distinct from Pydantic's runtime data validation |
| `.env` loading | `python-dotenv` | |
| Secrets scanning | `gitleaks` via `pre-commit` | blocks commits with secret-looking content; verified end-to-end (correctly blocks a real secret pattern, passes clean files) |
| Pre-commit file hygiene | `pre-commit-hooks` (trailing-whitespace, end-of-file-fixer, check-toml, check-added-large-files, check-merge-conflict) | standard baseline, low-effort to include |
| Generated-code validation | `ruff check` (Python), `sqlfluff` (SQL/dbt), `terraform validate` (Terraform), `dbt parse` (dbt) | dispatched by a `validate_code` tool in `coding_agent`, see Tool supply |

## Speed-first design decisions

(Deliberately lighter-weight than a correctness-graded system.)

- No strict answer-format evaluation/revision loop — that machinery
  exists only to satisfy exact-match grading, which doesn't apply here.
- Low tool-step caps (2-3, not 8+) — long tool-call chains feel broken
  in an interactive assistant.
- Supervisor routing decisions should be cheap/fast, not deep reasoning
  — a heuristic first, small-model classifier as fallback for ambiguous
  queries.
- Stream all output.
- Keep context windows capped (`num_ctx`) and history short unless a
  task specifically needs more.
- Tune Ollama `keep_alive` so the actively-used model stays warm between
  turns.

## Best practices to apply (learned from building a prior LangGraph agent)

- One job per agent/node — splitting research from answer-writing fixed
  real output-quality bugs previously; apply the same discipline here.
- Always set explicit step/iteration caps on any agentic loop from the
  start, not after it misbehaves.
- Tool descriptions matter as much as system prompts — models don't
  reliably pick the "obviously right" tool just because the prompt says
  to; write tool descriptions to be self-evidently correct.
- Test each agent node in isolation before wiring the full multi-agent
  graph together.
- Use an isolated project venv with pinned dependencies from day one —
  shared/dev environments can hide missing-dependency bugs.
- Use typed state (TypedDict or Pydantic) for the graph state schema.
- Accept that local models are meaningfully weaker than hosted models at
  multi-step tool orchestration — design for "good enough and fast,"
  not for matching hosted-model reliability.

## Sub-agents

- **Dev workflow**: use Claude Code forks/subagents to parallelize
  scaffolding once multiple independent files need writing (e.g. fork
  off `coding_agent.py` while designing `docs_agent.py` in the main
  thread), rather than always building strictly sequentially.
- **Architectural reference, not a v1 requirement**: Claude Code's own
  supervisor→subagent dispatch (async, results returned as events,
  context isolation so a sub-task's noise doesn't pollute the
  supervisor's context) is a real working instance of the pattern this
  project builds. Worth studying explicitly. v1 of this project's
  supervisor can stay synchronous/blocking for simplicity — a
  single-user local CLI doesn't need true async orchestration the way a
  background coding agent does; revisit only if blocking calls become a
  real responsiveness problem.

## Testing — two tiers, one command

Formalizes the "test each agent node in isolation" hard rule (already
in `CLAUDE.md`) into a real `pytest` suite, split by whether a test
needs live external services:

- **Directory structure mirrors the source tree** — `tests/test_agents/`
  mirrors `agents/`, `tests/test_tools/` mirrors `tools/`,
  `tests/test_rag/` mirrors `rag/`, etc. "Where's the test for X" maps
  directly to "where's X." See Suggested module structure below.
- **Component tests** (default) — fast, mocked dependencies, no live
  services. Covers the genuinely pure/deterministic logic: `safety.py`'s
  scope check and denylist matching, `validate_code.py`'s
  dispatch-by-file-extension, `rag/ingest.py`'s content-hash tracking
  and `*_BACKUP_*` exclusion pattern, the confidence-tier mapping,
  meta-command routing. Runs constantly, part of the normal dev loop and
  the `pre-commit` hook.
- **Live dependency tests** (opt-in, not a separate folder) — a small
  number of smoke tests against the real services this project depends
  on: Ollama reachable and returns a completion, `nomic-embed-text`
  produces an embedding, a value round-trips through Chroma, Tavily
  returns results for a query, self-hosted Langfuse is reachable. Marked
  tests can live in the *same file* as their mocked counterparts (e.g.
  `test_tools/test_web_search.py` has both) — the marker controls
  execution, not file placement. This directly targets the exact class
  of bug that bit the GAIA project — the `ddgs` vs `duckduckgo-search`
  package-name mismatch was invisible until tested against a real
  environment, not a mock.
- **Implementation**: `@pytest.mark.live` on the smoke tests, with
  `pytest` (no args) skipping them by default and `pytest -m live`
  running them explicitly. One test command, two speeds — not two
  separate tools to remember.
- **Not a pre-commit hook.** Live tests are slower and network-
  dependent; running them on every commit would fight the project's
  responsiveness priority. Run manually, or after changing a model
  name/dependency version/API key.

## Observability

Since this project is optimizing for responsiveness, measure it —
don't just assume it.

**Metrics to track:**

- **Latency**: time-to-first-token, total response time per turn,
  per-node/per-agent time breakdown (which node is the bottleneck),
  model cold-load penalty when Ollama swaps models.
- **Routing**: which agent was selected per query, number of
  agent-hops for chained multi-agent queries, heuristic-vs-classifier
  fallback rate.
- **Tool usage**: calls per query, success/failure rate, and which
  available tools are *never actually chosen* — a signal the tool's
  description needs work (this happened with `wikipedia_search` in the
  prior GAIA build).
- **RAG**: retrieval time, chunks retrieved vs. actually referenced in
  the final answer (cheap proxy for retrieval relevance).
- **System**: tokens/sec throughput per model, context length used per
  call (catches silent prompt bloat).

**Tooling — Langfuse from the start** (superseded the earlier "hand-roll
SQLite first, upgrade later" plan — decided to get a real live dashboard
immediately rather than build a throwaway text-only version first):

- **Self-hosted Langfuse** (Docker) — attach `CallbackHandler` to the
  compiled LangGraph graph (`.with_config(callbacks=[langfuse_handler])`,
  same pattern the reference GAIA agent example used) and most of the
  metrics above — latency per node, token counts, tool call
  inputs/outputs — are captured **automatically**, no custom timing
  wrapper needed. This is a real change of scope from the original plan:
  Langfuse's own callback hooks replace most of what `metrics.py` would
  have hand-rolled.
- **What still needs custom instrumentation** (not automatic from the
  callback handler, logged explicitly via the Langfuse SDK's
  score/event API — still all inside Langfuse, not a second system):
  which tools are *never chosen* (needs aggregation across sessions,
  not a per-call trace), RAG chunks-retrieved-vs-referenced, and the
  confidence tier assigned per response (see Confidence & validation).
- **`/stats`** keeps its original behavior — a live text summary printed
  in the terminal — but now sources that data from **Langfuse's API**
  (querying recent traces/sessions) instead of a hand-rolled SQLite
  store. One system of record (Langfuse), two ways to view it: the
  terminal for a quick glance via `/stats`, the browser dashboard for
  anything deeper.
- **Honest tradeoff, stated plainly**: this is a heavier operational
  piece than anything else in this project. Chroma and SQLite were
  chosen specifically because they need *no server*; self-hosted
  Langfuse runs as a multi-container Docker service (web app + Postgres,
  depending on version) that needs to actually be running for the app
  to trace anything. This is a deliberate, accepted departure from the
  "local-first, no server" pattern elsewhere — the payoff (a real live
  dashboard, decided over building one) was judged worth it.
- Needs `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in
  `.env` (generated by Langfuse's own UI on first setup) — same secrets
  handling as every other credential in this project.

## Suggested module structure

```
001_my_AI_assistant/
  .env.example                # placeholder secrets, committed (TAVILY_API_KEY, LANGFUSE_*)
  .gitignore                   # .env, __pycache__, .venv, etc — from commit #1
  .pre-commit-config.yaml       # gitleaks, ruff-check + ruff-format, file hygiene, mypy
  docker-compose.langfuse.yml    # self-hosted Langfuse (web app + Postgres)
  pyproject.toml                  # uv deps, ruff/pytest config, [project.scripts],
                                    # hatchling build pointing at src/myassistant
  src/myassistant/                 # src layout: everything namespaced under one package,
                                    # so `uv tool install .` doesn't put generic names
                                    # like `config`/`main` into site-packages
    __init__.py
    config.py                        # paths (ASSISTANT_HOME vs PROJECT_ROOT), models, keys
    main.py                           # REPL entry point + meta-commands + error handling
    supervisor.py                      # langgraph-supervisor setup: agents + routing
    state.py                            # shared graph state schema
    agents/
      coding_agent.py
      research_agent.py
      docs_agent.py
      general_agent.py
    tools/                                # LangChain tools, built from scratch
      safety.py                            # working-dir scope, confirmation gate, denylist
      validate_code.py                      # ruff/sqlfluff/terraform validate/dbt parse dispatch
    rag/
      ingest.py                              # chunk + embed + store notes (pypdf/python-docx/md)
      query.py                                # retrieve + generate
      memory.py                                # session summarization + conversation_memory
    observability/
      langfuse_client.py                       # CallbackHandler wiring + score/event calls
      stats.py                                  # /stats: text summary from Langfuse's API
  tests/                                        # mirrors the source tree - "where's the test for X" = "where's X"
    test_agents/
      test_coding_agent.py
      test_research_agent.py
      test_docs_agent.py
      test_general_agent.py
    test_tools/
      test_safety.py
      test_validate_code.py               # mocked + @pytest.mark.live tests can share a file
    test_rag/
      test_ingest.py
      test_query.py
      test_memory.py
    test_observability/
      test_langfuse_client.py
      test_stats.py
    test_supervisor.py
    test_state.py
```

## Suggested build order

0. **Repo scaffolding** — `uv init`, `pyproject.toml`, `.gitignore`
   (covering `.env` before anything else is committed), `.env.example`,
   `.pre-commit-config.yaml` with `gitleaks` wired in (add `ruff`/
   `pytest` hooks once there's code for them to check), `pytest`
   markers config (`live` marker, default run excludes it). Verify
   `.gitignore` + the `gitleaks` hook are both working *before* the
   first commit that touches any secret-adjacent code. Also stand up
   self-hosted Langfuse (`docker-compose.langfuse.yml`) and generate its
   API keys into `.env` — do this now, not later, since observability
   gets wired in as each agent is built, starting with step 2.
1. `main.py` — `prompt_toolkit`-backed REPL loop with meta-commands
   (`/help`, `/exit`) and their autocomplete, plus top-level error
   handling wired in from the start; no agent routing yet, one
   hardcoded response path, to prove the plumbing works end to end.
2. `langgraph-supervisor` wired to a single `research_agent` (fastest
   to stand up — reuses existing GAIA-project tools directly). **Also
   validate here** whether the small supervisor model (`llama3.2:1b/3b`)
   routes reliably — local models were inconsistent at tool-selection in
   the prior GAIA build, and `langgraph-supervisor` depends on reliable
   structured handoffs. If it's not reliable enough, size up the
   supervisor model before building further on top of it.
3. `coding_agent` with file/shell/git tools — build `tools/safety.py`
   (working-dir scope + confirmation gate + denylist) **before or
   alongside** the tools themselves, not after. Include
   `tools/validate_code.py` (`ruff check`/`sqlfluff`/`terraform
   validate`/`dbt parse` dispatch) as part of this agent's toolset from
   the start, not a later addition.
4. RAG pipeline (`ingest.py` + `query.py`) + `docs_agent`.
5. `general_agent` for fast low-stakes chat/drafts.
6. `rag/memory.py` — session summarization + `conversation_memory`
   collection, wired into session start/end in `main.py`; add the
   `/remember [text]` meta-command (on-demand save, same collection)
   as part of the same step, since it shares the summarization path.
7. Tune `langgraph-supervisor`'s routing (prompt/model choice) for
   ambiguous queries once all four agents exist.
8. `observability/langfuse_client.py` — attach `CallbackHandler` to the
   supervisor graph as soon as it exists (step 2), not bolted on at the
   end; add the custom score/event calls (tool-never-chosen, RAG
   relevance, confidence tier) as each of those features gets built.
9. `observability/stats.py` — `/stats` pulling a live text summary from
   Langfuse's API.
10. ~~Global CLI install~~ — **done early, during step 1.** The original
    plan deferred this to last, reasoning that moving paths late avoids
    revisiting prior steps. That inverted: defining `ASSISTANT_HOME` and
    `PROJECT_ROOT` correctly in `config.py` up front means never
    revisiting them, and adopting the `src/` package layout before there
    were files to move cost two files instead of twenty. Already
    verified: `uv tool install .` puts `myassistant` on `$PATH`, and
    `PROJECT_ROOT` correctly follows the launch directory. What remains
    for later steps is only pointing Chroma and the RAG manifest at
    `ASSISTANT_HOME` when those components get built (step 4).
