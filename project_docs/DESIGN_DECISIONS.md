# Important Design Considerations

Brief record of what changed from the first proposed architecture, and
why. See `PROJECT_REQUIREMENTS.md` for the current spec in full.

## Test directory structure — mirrors the source tree, not just tiers

- Originally: a flat `tests/unit/` + `tests/live/` split by test tier
  only. Revised on request: directories mirror the source tree
  (`test_agents/`, `test_tools/`, `test_rag/`, etc.) — "where's the test
  for X" now maps directly to "where's X." The unit/live distinction
  didn't go away, it just moved from folder placement to a `pytest`
  marker (`@pytest.mark.live`) — a mocked test and its live counterpart
  for the same module can live in the same file.

## Pre-commit hooks expanded beyond gitleaks + ruff

- Added the official `ruff-pre-commit` (version-pinned, matches the
  local `ruff` rather than whatever's on `$PATH`), standard
  `pre-commit-hooks` file hygiene (trailing whitespace, EOF fixer,
  TOML/large-file/merge-conflict checks), and `mypy` for static type
  checking — a deliberate addition alongside Pydantic, not a
  replacement: mypy checks source code type-consistency before
  anything runs, Pydantic validates actual data shape at runtime
  (specifically LangGraph state, per the existing typed-state hard
  rule) — different bugs, different timing, worth having both.
- **`gitleaks` was actually tested, not just configured and trusted.**
  Two early test attempts (an `AKIA...EXAMPLE` string, then a
  `sk-proj-...` OpenAI-format key) both passed silently — a real
  finding, not a false alarm: gitleaks' default `openai-api-key` rule
  only matches the *legacy* key format (`sk-` + 20 chars +
  `T3BlbkFJ` + 20 chars), not the newer `sk-proj-...` project-scoped
  keys OpenAI now issues by default. Also found that repeated-character
  filler text (`aaaa...`) fails the rule's entropy check even when the
  regex shape matches. Verified working end-to-end only once a
  correctly-patterned, sufficiently-random test secret was used — it
  was then correctly blocked (`exit code 1`) through the actual
  `pre-commit run` path, not just the raw `gitleaks` binary.

## REPL input — `prompt_toolkit` pulled forward from "later" to "now"

- Originally scoped as Tier 1, explicitly deferred: "add only if input
  history is later missed." Brought forward on explicit request for
  autocomplete — typing `/` shows a dropdown of meta-commands, and
  `/ingest <path>` gets filesystem-path completion.
- Deliberately narrow: this only changes the *input* side. Output stays
  plain streaming `print()` (Tier 0) — `rich`-style formatted output
  (Tier 2) is still deferred, and normal conversational input (not
  starting with `/`) gets no completion popup, keeping the common case
  fast and uncluttered.

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
- **"Later" arrived: external MCP servers adopted.** The interview-prep
  use case made the gap concrete — a ~3B model answering senior-level
  PySpark/Kafka/cloud questions produces fluent, specifically wrong
  detail, and nothing in the output signals it. Official docs are the
  grounding, and they cost no local RAM (network calls, not models).
  Microsoft Learn, Context7 and AWS Documentation are bound through
  `langchain-mcp-adapters`; Stack Exchange and GitHub REST are ordinary
  HTTP and stay hand-written. The original rule still holds where it
  matters: every tool *this project owns* is hand-written, and the
  adapter is only how someone else's server gets bound.
- **Adapter over hand-rolling the MCP protocol — a deliberate trade.**
  Hand-rolling the JSON-RPC handshake would have had more learning value
  (priority 2) and kept tool descriptions under our control. Chosen
  against because three servers means three handshakes, and the boilerplate
  repeats without teaching anything after the first. Two things are kept
  back from the adapter to preserve what mattered: the tool list is
  filtered before binding, and descriptions are rewritten rather than
  inherited.
- **Filtering is not optional.** AWS's documentation server alone exposes
  six tools; the three servers together are ~10. `get_tools()` returns
  them flat. Bound unfiltered onto `research_agent`'s existing two, this
  is the `wikipedia_search`-never-chosen failure at four times the scale.

## Agents vs. tools — the criterion, written down after nearly getting it wrong

- The question "why not one research agent per source — Azure, AWS,
  Spark?" came up, and the first answer given ("split by job, not by
  source") was too blunt to be useful. Replaced with a test that
  actually discriminates: **does the split change anything other than
  the tool list?** Different loop shape, evidence signal, model, or
  prompt → an agent. Only a different endpoint → a tool.
- Domain-splitting `research_agent` fails that test on every count: same
  model, same loop, same "did the fetch return real content" signal.
  It moves the decision from "which of N tools" to "which of N agents",
  both made by a ~3B model, so nothing is gained and a handoff is added.
- The concrete cost is cross-domain questions, which are normal for data
  engineering: "move data from Kinesis to Event Hubs" spans two clouds.
  One agent answers in one handoff with two tool calls; two split agents
  need two handoffs plus a synthesis node — and that node would both
  research and write, which the one-job rule forbids.
- arXiv is the counter-example that would earn its own agent: its loop
  is search → read abstract → judge relevance → maybe fetch, with a
  different evidence signal. Deferred, but on the right side of the line.
- An earlier draft of this advice proposed a dedicated `vendor_docs_agent`
  plus a keyword pre-filter *before writing any code* — premature
  optimisation against unmeasured routing failure, and inconsistent with
  the "start merged, split on evidence" position taken two paragraphs
  earlier. Corrected: add the tools to `research_agent`, and let the
  Langfuse tool-never-chosen metric decide whether a split is warranted.

## Getting `Observation` to the evidence gate — the string bottleneck

**The problem.** A LangChain tool's return value becomes a `ToolMessage`,
whose content is a string — so the structured `Observation` a tool builds
is flattened to text the moment it is returned. But the in-graph evidence
gate needs the object: `ok` and `metrics`, not prose. Two consumers, one
channel, and only one of them can have it.

Two common solutions:

1. **JSON serialization + parsing** — the tool returns JSON; the gate
   parses the `ToolMessage` content back into an `Observation`.
2. **Tool wrapper** — the wrapper calls the underlying function, gets the
   `Observation`, and returns a LangGraph `Command` carrying *both* the
   rendered string (as a `ToolMessage`) and the object (as a state update).

**Chosen: the tool wrapper.** JSON loses the thing this project cares most
about. The whole design rests on the observation *text* being the surface
where a model notices trouble — `[TOOL FAILED] ... status=403` is legible
to a 3B model in a way `{"ok": false, "detail": "...", "metrics": {...}}`
is not. Option 1 makes the gate's job easier by degrading the primary
safeguard to improve the secondary one, which is the wrong trade. The
variant that parses our own *rendered* output is worse still: the gate
would depend on a string format, and the gate exists precisely because
reading strings is unreliable.

The wrapper preserves what `Observation` was built for — `render()` for
the model, `ok`/`metrics` for the graph — instead of collapsing them back
into one channel.

**Verified, not assumed**, against the pinned langgraph: a tool returning
`Command(update={"observations": [...], "messages": [ToolMessage(...)]})`
is passed through by `ToolNode`, and the `operator.add` reducer on
`observations` merges it. Needs `InjectedToolCallId` — which lives in
`langchain_core.tools`, not `langgraph.prebuilt`, in this version.

**Rejected third option:** a per-turn collector (module global or
`ContextVar`) that tools append to. Simpler, but it is hidden mutable
state outside the graph, invisible to LangGraph's checkpointing — and it
would fight the planned `asyncio.gather` over concurrent tool calls,
which is the one place parallelism actually pays here.

**Cost accepted:** each tool needs a thin wrapper rather than being a
plain function, and the tools are now coupled to LangGraph's `Command`
API rather than being framework-neutral. Worth it — the underlying
functions stay pure and independently testable, which is where the
existing tool tests already operate.

## Routing checkpoint — measured, and the supervisor model changed

Ran the step-2 checkpoint the spec asks for: real Ollama, stub agents,
8 questions, scoring which agent got the first handoff.

| Supervisor model | Correct first hop | Supervisor answered itself |
|---|---|---|
| `llama3.2:3b` | 6/8 (5/8 on a rerun) | 7/8 |
| `qwen2.5:3b-instruct` | **8/8** | 8/8 |

- **Switched `SUPERVISOR_MODEL` to `qwen2.5:3b-instruct`.** llama3.2:3b
  was unstable run to run, and sometimes did not hand off at all — it
  just answered the question, which is the thing the architecture exists
  to prevent. Costs about 1.2s more per turn.
- **A bonus we did not plan for:** `RESEARCH_MODEL` was already
  qwen2.5:3b, so supervisor → research is no longer a model swap. One 3B
  model in memory (1.93GB) instead of two (3.95GB), and `llama3.2:3b`
  is now unused.
- `CRITIC_MODEL` and `GENERAL_MODEL` alias `SUPERVISOR_MODEL`, so they
  moved too. Both are fine on qwen.
- **A measurement lesson:** the first run of this used stub agents that
  replied "Grounded answer." The supervisor kept re-routing, because a
  useless answer gives it nothing to stop on. Realistic stub replies
  changed the numbers. A bad harness can look like a bad model.

## Supervisor writes its own answer — known, deliberately left alone

- After an agent hands back, `langgraph-supervisor` calls the supervisor
  again, and it writes a final message of its own. So the user sees the
  agent's answer followed by the supervisor's paraphrase of it.
- That breaks the one-job rule. It is **structural to the library, not a
  model failure** — both models did it, 7/8 and 8/8.
- **Left in place on purpose for now**, so it can be seen first-hand in
  the REPL before being designed around. The likely fix when we get
  there is to display the last *agent* message rather than the last
  message; the full trace stays in Langfuse either way.

## Things that bit us while wiring step 2

Self-contained on purpose — this section is meant to be lifted into its
own file later.

**`langchain-mcp-adapters` almost did not fit.** One pin cascades into
another: Langfuse v2 needs LangChain 0.3.x, and the newer adapters need
LangChain 1.x. Tested three versions:

| Version | Result |
|---|---|
| 0.2.1+ | requires `langchain-core>=1.0.0` outright |
| 0.2.0 | *claims* to accept 0.3.x, but imports `langchain_core.messages.content`, which only exists in 1.x — the declared floor is simply wrong, and it fails at import |
| **0.1.14** | actually works |

Also had to add `mcp<2`. The adapter leaves `mcp` unbounded, so a fresh
install picks 2.x, where `mcp.shared.context` no longer exports
`RequestContext` — import error again. Lesson: a package's declared
dependencies are a claim, not a guarantee. Import it and see.

**Wiring the supervisor quietly made the test suite hit live Ollama.**
`test_main.py` calls `run_turn`, `run_turn` calls `answer`, and `answer`
now builds the real graph. A 0.7s suite became 26s of real model calls,
and it broke the rule that live services are opt-in only. Nothing
failed, which is why it was easy to miss — the tests still passed, just
slowly, and against a live model. Fixed with an autouse fixture that
stubs the graph and the Langfuse client. Lesson: when a leaf function
starts reaching outside the process, every test above it does too.

## Cross-role questions retrieve too narrowly — solved by knowing the roles

"Walk me through my career" returned 4 chunks from 3 files and omitted
two employers. Top-k ranks by similarity alone, so the best-matching
document's near-identical chunks crowd out other sources. The tier still
said HIGH: it certifies retrieval *quality*, not *coverage*.

Restructuring storage does not fix it. Measured: the cross-role documents
are 10k-125k characters, so no chunking scheme returns "your whole
career" into a 3B context, and a collection per employer would turn one
retrieval into N plus a routing decision — harder for a small model, not
easier.

What does fix it is knowing the roles up front. Chunks are tagged with a
`role` at ingest (inferred from the path, which already encodes it), and
`search_across_roles` runs one filtered search per role. Coverage becomes
structural rather than hoped for, and roles are returned in configured
priority order so a truncated answer loses the least.

The roles themselves live in `.env`, not in the repo. Employment history
is not secret, but it is personal, and this repo is public — the same
split already used for credentials, extended from "secret" to "personal".
The repo ships the mechanism; the machine supplies the data.

## Vision model chosen by measurement, not by size

Three models were tested against a diagram whose contents were verified
first, which mattered more than usual here.

| Model | Size | Result |
|---|---|---|
| moondream | 1.7GB | Fabricated a whole plausible schema. Named a `salesperson` table with email, phone and zipcode columns; none exist. |
| granite3.2-vision:2b | 1.5GB | Read it correctly under one prompt, claimed blindness under another. 16s-964s for the same image. |
| **qwen2.5vl:3b** | 3GB | 9/9 on both a specific and a generic prompt, 13-36s. Ships. |

- **Downscaling is the whole trick.** The same architecture diagram gave
  35 useless characters at 4843x5796 and at 2898x2421, then 2049
  characters of real content at 1600px. Total pixel count is what breaks
  it, not fine detail — so tiling was tried and is not needed.
- **They fail silently.** An unreadable image returns "The image appears
  to be a flowchart": fluent, confident, empty, and not an error. Caught
  by a length-and-phrase check, the same job `looks_empty()` does for a
  redirect-shell web page.
- **Prompt phrasing mattered enormously for the weak models and not at
  all for the good one.** granite scored 5/7 on "list the tables and
  columns" and 0/7 on "read the contents of the image"; qwen scored 9/9
  on both. A model that only works when you already know what the image
  contains is no use for the case where you don't — which is the case.
- The vision model unloads after two minutes. It runs only when an image
  is involved, and keeping 3GB resident would evict the supervisor, which
  runs on every turn.

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

## Evaluation reinstated — as a loop step, and as a measurable flag

Reverses part of the early "no evaluate/revision loop" decision above.
Two separate changes, for different reasons:

- **A first pass framed this as a fourth `evaluate` step the agent
  performs. That was wrong, and correcting it is the useful part.**
  Inside the model there is no separate evaluation function: the loop is
  `reason → act → observe → reason`, and what reads as evaluation is the
  reasoning step running again with the observation now in context —
  next-token prediction over a richer history, not a check being
  computed. (That it tends to catch mismatches at all is a product of
  post-training, not of an architectural component.) Prompting for an
  explicit "now evaluate" phase would produce text that *looks* like
  scrutiny while adding nothing the next reasoning step didn't already
  have.
- **So the leverage moved from prompting to tool return values.** The
  model's self-assessment is bounded by what is legible in the
  observation text; it has no privileged access to "did this work." A
  tool that fails quietly and returns something plausible leaves nothing
  to notice, and the model then reports success *correctly*, given its
  context. The step cap doesn't help — an agent that believes it
  succeeded stops looping, so the cap never binds. Hence
  `tools/observation.py`: loud failure markers, numbers on every result,
  and `fetched()` routing claimed successes through `looks_empty()` so a
  200-with-a-consent-wall becomes an explicit failure. This is the same
  reasoning that killed the prompt-only "must call `visit_webpage`" rule
  one section above, applied to control flow instead of labeling.
- **The deterministic gate belongs in the graph for the same reason.**
  It is not "the agent evaluating itself" — it is the graph checking
  facts the model cannot see, which is exactly why it must not depend on
  the model having noticed them.
- **A model critic, off by default behind `CRITIC_ENABLED`.** This one
  genuinely does contradict priority 1, and was adopted anyway on an
  explicit rationale: *the latency claim should be measured, not
  assumed*. The project already argues this about everything else —
  "since this project is optimizing for responsiveness, measure it,
  don't just assume it" — and the flag makes the A/B a one-variable
  change with Langfuse spans on both sides. If the numbers say it's
  expensive, that's a finding, and the default stays off.
- Two constraints keep the critic from becoming the thing the original
  ban was right about: it reuses `SUPERVISOR_MODEL` (always resident, so
  the measurement isn't polluted by a 2-5s model swap), and it is handed
  the tool observations rather than only the answer text — "is this
  supported by the evidence" is answerable, "is this correct" is not.
- **Stated plainly so the measurement isn't over-read:** the critic
  shares the agent's blind spot. Reading the same observation text, it
  cannot detect a silent tool failure any better than the agent did. Its
  edge is only a fresh context, free of the agent's committed narrative.
  If the latency numbers come back bad, the observation contract is
  where the real defect-catching lives anyway.

## Self-reported confidence — banned from the screen, allowed in the log

- The original rule said confidence is "never a raw model self-report."
  Read literally that also forbids *measuring* the self-report, which
  was never the intent — the objection is that a printed "87% confident"
  reads as far more rigorous than it is. Displaying it and recording it
  are different acts.
- So `SELF_REPORT_ENABLED` (default off) logs the model's own number to
  Langfuse under a separate score name. Two names, never merged:
  merging would launder a guess into the evidence channel, and comparing
  the two is the whole point.
- This is the same reasoning as the critic flag — the project asserts
  that self-reports are uncalibrated, and now it can check that against
  its own models instead of taking it on faith. The expected result
  (0.85-0.95 regardless of what the tools actually returned) would make
  the case for the rule far better than the assertion does.
- Normalising to 0-1 is not tidiness: small models answer "90" where 0.9
  was asked for, sometimes within a single run, and a mixed series is
  unanalysable.
- **Wiring constraint for step 2:** the number must come from a field on
  the structured output the agent already produces — never a second
  model call. A separate "now rate your confidence" round trip would
  cost full latency to collect a number the experiment exists precisely
  because we don't trust yet. If it can't be had as a field on the
  existing call, it isn't worth having.

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

## `src/` package layout — flat modules would have broken the global install

- The spec's module structure put `main.py`, `config.py`, `state.py`,
  `agents/`, `tools/` at the repo root. That works when running
  `python main.py`, but `uv tool install .` would then install `config`,
  `main`, `state`, `agents` and `tools` as top-level names in site-packages —
  generic enough to collide with other packages, and `[project.scripts]`
  needs a real importable package path anyway.
- Moved to `src/myassistant/`, everything namespaced under one package, with
  `myassistant = "myassistant.main:main"` as the entry point.
- `src/` specifically (rather than `myassistant/` at the repo root) so tests
  import the *installed* package rather than the adjacent source directory —
  otherwise the suite can pass against code that isn't actually installable.
  Consequence: `pythonpath = ["."]` came back out of the pytest config.
- Done at step 1, when it cost two files to move. Deferring it to step 10 as
  originally sequenced would have meant rewriting every import in the project.

## `config.py` — exists for safety correctness, not for tidiness

Not in the original module structure; added when the global-install work made
"which directory am I allowed to write to?" a question with two possible
answers. Worth stating plainly, because the file looks like a settings bucket
and its real job is narrower than that.

- **The bug it prevents.** `PROJECT_ROOT` is the fence around `coding_agent` —
  the only directory its file/shell/git tools may touch. If each module called
  `Path.cwd()` at the moment it needed it, that fence would move whenever
  anything changed directory mid-session: a build script, a test using
  `monkeypatch.chdir`, a `cd` inside a shell command. The agent would then
  write outside the directory you launched in **and the safety check would
  still report "passed,"** because it would be comparing against the new,
  wrong directory. Capturing `Path.cwd()` exactly once at import makes that
  failure unrepresentable rather than merely unlikely.
- **The two "homes" must not be conflated.** `ASSISTANT_HOME`
  (`~/.myassistant`) is fixed across launches and holds the knowledge base;
  `PROJECT_ROOT` changes every launch. If the knowledge base were
  `cwd`-relative, ingested notes would scatter across every directory the tool
  was ever run from — the failure mode the "two kinds of state" requirement
  was written to prevent.
- **Load ordering.** `load_dotenv()` has to run before any module reads a key.
  A single module that executes first guarantees that; scattered `load_dotenv`
  calls do not.
- **Model names live here, not in `.env`** — a deliberate split by *kind of
  setting*, not by convenience. Credentials go in `.env` (gitignored). Model
  names, step caps and `keep_alive` values are non-secret and part of the
  project's reproducible behavior, so they belong in version control, in the
  same category as `line-length = 100`. Putting them in `.env` would mean a
  fresh clone starts with nothing configured and the values never appear in
  code review. They stay overridable via environment variable, and
  `.env.example` lists them commented-out so they are discoverable without
  reading source.
- **Rejected: a `[tool.myassistant]` section in `pyproject.toml`.** Reads
  nicely, but once `uv tool install .` puts this on `$PATH`, the installed CLI
  cannot reliably read its own `pyproject.toml` at runtime — it would break
  the global-install step for no real gain.
- **Deviation from the documented build order**, made deliberately: the spec
  sequenced "move global-persistent state to `~/.myassistant/`" as step 10,
  reasoning that moving paths earlier means revisiting prior steps. That
  reasoning inverts — defining the paths correctly in step 1 means never
  revisiting them, and leaves step 10 as packaging only. Consequence:
  `manifest.db` now lives in `ASSISTANT_HOME`, not `rag/`, so the spec's
  module structure and the stale `rag/manifest.db` line in `.gitignore` both
  need updating.

## LangChain pinned to 0.3.x — forced by the Langfuse v2 SDK, found by testing imports

- The spec named libraries but no versions. Left unpinned, `uv` resolved to
  LangChain 1.x / LangGraph 1.x — and **Langfuse v2's LangChain integration
  does not work on 1.x**: it imports `langchain.callbacks.base`, a module
  1.x removed, so `from langfuse.callback import CallbackHandler` raises
  `ModuleNotFoundError`. Observability would have been dead on arrival.
- Caught by running an import smoke test immediately after `uv sync` rather
  than assuming resolution implies compatibility — the same class of bug as
  the GAIA project's `ddgs`/`duckduckgo-search` mismatch, and the reason the
  live-test tier exists.
- The two constraints are linked and neither is free: Langfuse v2 was pinned
  because the v3 server needs ClickHouse + Redis + MinIO (~4-6GB), which does
  not fit in 16GB alongside Ollama. Holding that pin forces LangChain 0.3.x.
  Resolved set: `langchain 0.3.30`, `langgraph 0.6.11`,
  `langgraph-supervisor 0.0.29`, `langfuse 2.60.10`.
- Accepted tradeoff: 0.3.x is a maintenance branch rather than the current
  line. It matches what this project's spec and the prior GAIA build were
  written against, so those patterns transfer directly — but this is a
  deliberate deferral, not a permanent choice. Upper bounds in
  `pyproject.toml` carry a comment explaining *why*, so a future version bump
  doesn't silently re-break tracing.

## Models — lighter set than the spec's first-listed option, chosen on memory not quality

- The spec listed alternates for two roles without picking. Chose
  `qwen2.5-coder:7b` (coding), `qwen2.5:3b-instruct` (research/docs),
  `llama3.2:3b` (supervisor), `nomic-embed-text` — ~8.9GB.
- The deciding constraint was **concurrent residency, not inference speed**.
  A single 7B model runs fine on an M4; the problem is that a chained query
  (supervisor → specialist → specialist) touches three models, and two 7Bs
  plus macOS plus Docker exceeds 16GB — causing eviction and a 2-5s cold-load
  penalty *per agent hop*, directly against the responsiveness priority.
- Coding stays at 7B because it is the primary use case and most sensitive to
  a quality drop. Research/docs took the 3B cut because that agent mostly
  summarizes retrieved text rather than reasoning from scratch — the cheapest
  place to lose capability. Supervisor uses `llama3.2:3b` not `:1b`; 1B is
  unreliable at the structured handoffs `langgraph-supervisor` depends on.

## Global CLI install — launch-anywhere modeled on Claude Code, not in the original design

- Original design assumed the assistant ran from inside its own repo
  directory (`python main.py` or `uv run main.py`), with no distinction
  between the assistant's install location and whatever directory the
  user actually wanted `coding_agent` to work in.
- Prompted by wanting Claude-Code-like ergonomics: launch from any
  directory on the machine and have it work there, including
  `coding_agent` operating on that directory's contents.
- Resolved by splitting state into two explicit tiers rather than
  adding a special case: **global/persistent** (Chroma store,
  `rag/manifest.db`, `.env`, Langfuse config — moves to
  `~/.myassistant/`, independent of launch location) vs.
  **per-launch/`cwd`-scoped** (`coding_agent`'s existing working-directory
  safety boundary, now explicitly defined as `Path.cwd()` at launch
  time). This wasn't a new capability for `coding_agent` — the safety
  boundary already existed — just a correction that "project root" in
  that boundary means the launch directory, not the assistant's own
  repo.
- Other agents (`research_agent`, `docs_agent`, `general_agent`) don't
  get their own separate file-access logic for repo content — they
  route through `coding_agent`'s already-scoped tools via supervisor
  chaining, keeping one safety boundary instead of several.
- Installation mechanism: `uv tool install .`, which needs a
  `[project.scripts]` entry in `pyproject.toml` and a real `main()`
  function (`main.py` currently is still `uv init`'s placeholder stub).
  Sequenced last in the build order (step 10) — moving global-persistent
  paths only makes sense once every component that touches them already
  exists, otherwise every prior step would need revisiting.

## `/remember` — on-demand memory save, added alongside the launch-anywhere change

- The original `conversation_memory` design only had one write path:
  automatic summarization at `/exit` or process end. That meant a
  useful moment mid-conversation couldn't be persisted without ending
  the session.
- Added `/remember [text]` as a meta-command: no arguments summarizes
  the conversation so far (same cheap model and path as end-of-session
  summarization); given text, stores it verbatim with no summarization
  step, since the user has already distilled it themselves.
- Deliberately narrow scope: always writes to `conversation_memory`
  only, never `tech_notes` or `resume_interview` — those already have
  their own deliberate ingestion path (`/ingest`) and mixing an ad hoc
  meta-command into that path would blur the two.

## Sub-agents — scoped down from an early broader idea

- Clarified into two separate, smaller things rather than one big
  architectural commitment: using Claude Code's own forking as a *dev
  workflow* convenience (parallelizing scaffolding), and treating Claude
  Code's supervisor→subagent dispatch as an *architectural reference*
  to study, not a v1 requirement — this project's supervisor stays
  synchronous/blocking for simplicity, revisited only if that becomes a
  real responsiveness problem.


Using langchain-mcp-adapters:
langchain-mcp-adapters is an official LangChain library that bridges MCP servers into LangChain's (and LangGraph's) own tool-calling system, so you can plug MCP tools straight into a LangChain agent without writing custom glue code yourself.

Here's the actual problem it solves: LangChain has its own internal notion of what a "tool" is — a Python object with a name, description, an input schema, and a callable function, structured a specific way its agents know how to work with. MCP servers, on the other hand, expose tools through the MCP protocol itself — you connect a client session, call list_tools(), get back MCP-shaped tool definitions, and invoke them via call_tool(), all as JSON-RPC messages over stdio or HTTP/SSE. Those two shapes aren't compatible out of the box. Without an adapter, you'd have to manually loop over whatever an MCP server's list_tools() returns and hand-wrap each one into a LangChain Tool object yourself, translating the schema and wiring up the call/response plumbing in both directions.
