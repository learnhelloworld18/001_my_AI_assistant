# Session Handoff

Where the build is, for picking up in a new session. Read `CLAUDE.md` first
(rules), then this (state). Last updated during step 2.

## Done

**Step 0 — scaffolding.** All verified working, not just configured:

- Ollama: `qwen2.5-coder:7b-instruct-q4_K_M` (4.68GB), `qwen2.5:3b-instruct`
  (1.93GB), `llama3.2:3b` (2.02GB), `nomic-embed-text` (0.27GB). Lighter than
  the spec's first option — two 7B models plus macOS plus Docker will not fit
  in 16GB, and eviction costs 2-5s per agent hop.
- Deps installed and import-smoke-tested. **LangChain pinned to 0.3.x**: the
  Langfuse v2 SDK imports `langchain.callbacks`, removed in 1.x, which
  silently kills all tracing.
- Langfuse v2 running in Docker (~220MB, `mem_limit` guardrails set). Pinned
  to v2 because v3 needs ClickHouse/Redis/MinIO.
- `.env` complete and verified: Tavily returns live results, Langfuse
  `auth_check()` passes and a test trace was received.

**Step 1 — REPL.** `config.py` + `main.py`, all 9 pre-commit hooks green.

- `config.py`: `ASSISTANT_HOME` (global, `~/.myassistant/`) vs `PROJECT_ROOT`
  (`Path.cwd()` captured once at import). The second is `coding_agent`'s future
  fence — resolving it once means it cannot drift if anything `chdir`s.
- `main.py`: `COMMANDS` registry drives both dispatch and autocomplete so they
  cannot diverge; completion fires only on `/`; per-turn error handling means
  one bad turn never kills the loop.
- `Session` holds `history`, `session_id` (groups Langfuse traces),
  `started_at`. `/clear` wipes history but keeps the id.

**Step 10 — done early.** `src/` layout plus `[project.scripts]`, so
`uv tool install .` puts `myassistant` on `$PATH` and it launches from any
directory.

**Step 2 — in progress.** Three files landed, 55 tests passing:

- `tools/observation.py` — the contract every tool returns. `Observation`
  (frozen), `fetched()`, `failed()`, `looks_empty()`. Failure leads with
  `[TOOL FAILED]`, success carries its numbers, and `fetched()` downgrades a
  200-with-a-consent-wall to an explicit failure.
- `observability/langfuse_client.py` — `get_callbacks()`, `score()`,
  `score_self_report()`, `flush()`, `reset()`. Every failure mode degrades to
  an empty callbacks list; that empty list is the no-op.
- `state.py` — `AssistantState` (TypedDict, `total=False`), `ConfidenceTier`,
  `Verdict` (Pydantic), `tier_from_observations()`, `render_evidence()`.

## Next: finish step 2

Remaining files: `tools/web_search.py`, `tools/visit_webpage.py`,
`agents/research_agent.py`, `supervisor.py`, `agents/critic.py`, plus wiring
`main.answer()` to the supervisor and `flush()` into `/exit`.

- Reuse `visit_webpage.py` from
  `/Users/him/learn-C-One/hugging_face_agents_course/final_challenge_GAIA_benchmark/tools/`
  — keep its `query` parameter. The real change is wrapping its return in
  `fetched()`; it currently hands back bare strings. `web_search.py` there is
  DuckDuckGo-based and needs rewriting for Tavily.
- **Checkpoint the spec calls for:** verify `llama3.2:3b` produces reliable
  structured handoffs. Local models were inconsistent at tool-selection in the
  prior GAIA build. If routing is flaky, size the supervisor up *before*
  building three more agents on top of it — note that also moves
  `CRITIC_MODEL`, which is aliased to `SUPERVISOR_MODEL`.

**Then, after the checkpoint:** external documentation sources (Microsoft
Learn, then Context7, then AWS) via `langchain-mcp-adapters`, plus Stack
Exchange and GitHub REST as hand-written `@tool`s. See PROJECT_REQUIREMENTS'
"External documentation sources" — filtering the tool list before binding is a
hard requirement, not a nicety.

Then steps 3-9 per `PROJECT_REQUIREMENTS.md` (build order). Note the
interview-prep use case argues for **step 4 (RAG) before step 3
(coding_agent)** — grounding technical answers matters more than code tools.

## Decisions already settled — don't relitigate

Full reasoning in `DESIGN_DECISIONS.md`. Briefly:

- Model names live in `config.py`, not `.env` — they are not secrets, and
  belong in version control. Only credentials go in `.env`.
- `history` is `list[tuple[str, str]]` today and becomes `list[BaseMessage]`
  when `main.answer()` is wired to the supervisor. `state.py` did not force it.
- No `ended_at` on `Session`; `PromptSession` stays out of `Session`.
- **There is no fourth "evaluate" step inside the model.** Evaluation is the
  reasoning step running again with the observation in context. The leverage is
  observation text, not prompting. Three layers: in-model reasoning, the
  deterministic in-graph evidence gate, and the optional critic.
- **`CRITIC_ENABLED` and `SELF_REPORT_ENABLED` are both off by default.** Both
  exist to measure a claim rather than assume it. The critic shares the agent's
  blind spot; the self-report is logged, never displayed.
- **Agents vs. tools:** split into a new agent only when the loop shape,
  evidence signal, model or prompt differs — not when only the data source
  does. Azure/AWS/Context7 are tools on `research_agent`, not separate agents.
- **Four agents total** (`coding_agent`, `research_agent`, `docs_agent`,
  `general_agent`) plus the supervisor and the optional critic node. External
  doc sources do not add agents.
- Ollama config stays as-is. A second Ollama instance duplicates model weights
  for no throughput gain on one GPU; `OLLAMA_MAX_LOADED_MODELS` is the lever if
  eviction ever shows up in Langfuse.

## Gotchas found the hard way

- **`pre-commit` stashes untracked/unstaged files and restores them.** This
  very file went missing mid-session that way and had to be recovered from a
  dangling git blob (`git fsck --unreachable` → `git cat-file -p`). Commit
  before running hooks on a dirty tree.
- **`pre-commit run --all-files` only checks git-tracked files.** After moving
  files it reported "no files to check" for ruff and mypy while genuinely
  failing code sat unstaged. `git add` first, then trust the result.
- **`langfuse.auth_check()` raises `ConnectError` when the container is down**
  — it does not return `False`. Bad keys return `False`; an unreachable host
  raises. Both paths must be handled or the REPL crashes on the common case.
- **`from __future__ import annotations` turns state annotations into
  ForwardRefs.** LangGraph still resolves the reducers correctly, but tests
  must assert on `StateGraph(...).channels`, not `__annotations__`.
- **mypy sees `MessagesState` as `Any`** (langgraph ships no stubs), so
  `total=False` on a subclass is rejected. `AssistantState` spells the schema
  out instead; same resolved channels.
- **mypy is not in the dev dependency group** — it only exists inside
  pre-commit's isolated env, so `uv run mypy src` fails with "Failed to spawn".
  Use `uv run pre-commit run mypy`.
- `.gitignore` needs `.env.*` not just `.env` — a `.env.bak` with live secrets
  showed up untracked and would have been caught only by gitleaks.
- Anything reading `os.environ` must import `myassistant.config` first;
  `load_dotenv()` runs there. `langchain-tavily` reads the key at init.
- There is **no `requirements.txt`** — dependencies live in `pyproject.toml`
  plus `uv.lock`. `CLAUDE.md` still refers to `requirements.txt`; that
  reference is stale.

## Verify the environment still works

```bash
cd /Users/him/learn-C-One/001_my_AI_assistant
uv run pytest tests/ -q                                    # 55 passing
uv run pre-commit run --all-files                          # 9 hooks
docker compose -f docker-compose.langfuse.yml ps           # Langfuse up
curl -s localhost:11434/api/version                        # Ollama up
```

`.env` is gitignored, so a fresh clone needs `cp .env.example .env` and real
values before anything works.
