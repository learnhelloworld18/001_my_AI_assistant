# Session Handoff

Where the build is, for picking up in a new session. Read `CLAUDE.md` first
(rules), then this (state). Last updated after step 1.

## Done

**Step 0 — scaffolding.** All verified working, not just configured:

- Ollama: `qwen2.5-coder:7b-instruct-q4_K_M`, `qwen2.5:3b-instruct`,
  `llama3.2:3b`, `nomic-embed-text` (~8.9GB). Lighter than the spec's first
  option — two 7B models plus macOS plus Docker will not fit in 16GB, and
  eviction costs 2-5s per agent hop.
- Deps installed and import-smoke-tested. **LangChain pinned to 0.3.x**: the
  Langfuse v2 SDK imports `langchain.callbacks`, removed in 1.x, which
  silently kills all tracing.
- Langfuse v2 running in Docker (~220MB, `mem_limit` guardrails set). Pinned
  to v2 because v3 needs ClickHouse/Redis/MinIO.
- `.env` complete and verified: Tavily returns live results, Langfuse
  `auth_check()` passes and a test trace was received.

**Step 1 — REPL.** `src/myassistant/config.py` + `main.py`, 17 tests, all 9
pre-commit hooks green.

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
directory. Flat top-level modules would have installed generic names
(`config`, `main`, `tools`) into site-packages.

## Next: step 2 — supervisor + research_agent

Files: `state.py`, `tools/web_search.py`, `tools/visit_webpage.py`,
`agents/research_agent.py`, `supervisor.py`,
`observability/langfuse_client.py`, plus tests.

- Reuse `visit_webpage.py` from
  `/Users/him/learn-C-One/hugging_face_agents_course/final_challenge_GAIA_benchmark/tools/`
  — it works nearly as-is, keep its `query` parameter. `web_search.py` there is
  DuckDuckGo-based and needs rewriting for Tavily.
- Replace `main.answer()` with the supervisor call.
- `langfuse_client.py` needs a no-op fallback so a stopped container degrades
  to "no tracing" rather than crashing the REPL.
- **Checkpoint the spec calls for:** verify `llama3.2:3b` produces reliable
  structured handoffs. Local models were inconsistent at tool-selection in the
  prior GAIA build. If routing is flaky, size the supervisor up *before*
  building three more agents on top of it.

Then steps 3-9 per `PROJECT_REQUIREMENTS.md` (build order).

## Decisions already settled — don't relitigate

Full reasoning in `DESIGN_DECISIONS.md`. Briefly:

- Model names live in `config.py`, not `.env` — they are not secrets, and
  belong in version control. Only credentials go in `.env`.
- `history` is `list[tuple[str, str]]` today and will likely become
  `list[BaseMessage]` at step 2, because a supervisor turn emits several
  messages. Left alone until step 2 forces the shape.
- No `ended_at` on `Session`: it would be `None` for the object's whole life
  except the final instant. Step 6 can call `datetime.now(UTC)` directly.
- `PromptSession` stays out of `Session`. One is I/O needing a TTY, the other
  is data that step 6 serializes; merging them would drag terminal setup into
  every test.

## Gotchas found the hard way

- **`pre-commit run --all-files` only checks git-tracked files.** After moving
  files it reported "no files to check" for ruff and mypy while genuinely
  failing code sat unstaged. `git add` first, then trust the result.
- `.gitignore` needs `.env.*` not just `.env` — a `.env.bak` with live secrets
  showed up untracked and would have been caught only by gitleaks.
- Anything reading `os.environ` must import `myassistant.config` first;
  `load_dotenv()` runs there. `langchain-tavily` reads the key at init.

## Verify the environment still works

```bash
cd /Users/him/learn-C-One/001_my_AI_assistant
uv run pytest tests/ -q                                    # 17 passing
uv run pre-commit run --all-files                          # 9 hooks
docker compose -f docker-compose.langfuse.yml ps           # Langfuse up
curl -s localhost:11434/api/version                        # Ollama up
```

`.env` is gitignored, so a fresh clone needs `cp .env.example .env` and real
values before anything works.
