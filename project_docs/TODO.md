# TODO

Backlog. `implementation_steps.md` records what is done; this records what
isn't. Move items out as they land.

## Drag-and-drop a file into the REPL

Drag a file onto the terminal, the REPL recognises the pasted path, asks
permission to read it, and then can use it for the conversation.

What has to happen:

1. **Recognise the paste.** Dragging into a macOS terminal pastes a
   *shell-escaped* path — spaces become `\ `, or the whole thing arrives
   single-quoted. So the raw input is not a usable path until unescaped.
   Detection rule: after unescaping, does it resolve to an existing file?
   If yes, treat it as a file, not a question.
2. **Ask before reading.** A short confirmation showing the resolved
   absolute path and size, defaulting to no. Same principle as the
   `coding_agent` confirmation gate.
3. **Then read it** and make the content available to the turn.

### The decision this needs first

**A dragged file is almost always outside `PROJECT_ROOT`** — that is the
point of dragging it. But `PROJECT_ROOT` is `coding_agent`'s hard fence,
and "no arbitrary absolute paths, no traversal outside that root" is a
non-negotiable in `PROJECT_REQUIREMENTS.md`.

So this feature is a deliberate, user-confirmed exception to that fence,
and it needs scoping before it is built:

- Does confirming grant a **one-time read**, or add the path to a
  **session allowlist** so follow-up questions can re-read it?
- Is the exception **read-only**? It should be. Reading is the safe half
  of the boundary; writing outside `PROJECT_ROOT` stays forbidden.
- Does the denylist still apply? Yes — a dragged `.env`, `.pem`, or
  `~/.ssh/id_rsa` must be refused regardless of confirmation. Otherwise
  drag-and-drop becomes a way to walk around the credential rule.

### And one more, before writing code

**Is this the same feature as `/ingest`, or a different one?** They look
similar and are not:

- `/ingest <path>` — store permanently in Chroma, for every future session.
- drag-and-drop — read *this file, for this conversation*, then forget it.

Both are wanted. Deciding which a dragged file means (or asking) is part
of the design, not an afterthought.

## Known, from step 2

- **Supervisor talks around the agent's answer**, both before the handoff
  ("Let me fetch that now") and after it (a full paraphrase). Streaming
  made this much more visible — a research question now shows the answer
  essentially twice. Left in on purpose; see DESIGN_DECISIONS. The fix is
  to print only tokens from agent subgraphs, which streaming has made easy:
  the namespace on each chunk already says which subgraph it came from.
- **`agents/critic.py`** — the last piece of the evaluation loop, off by
  default behind `CRITIC_ENABLED`.
- **MCP doc sources** — Microsoft Learn, then Context7, then AWS. The
  adapter is installed and pinned; nothing written yet.
- **Step 4 (RAG) before step 3 (`coding_agent`)?** The interview-prep use
  case argues yes — grounding technical answers matters more than code
  tools. Not yet decided.

- **Genericise remaining personal examples** in `query.py` and
  `docs_agent.py` docstrings, and in `PROJECT_REQUIREMENTS.md`. The data
  itself now lives in `.env`, but a few illustrative strings still name
  real employers and files. Cosmetic; the repo is public.
- **A one-page prose career summary** in the documents folder. Prose
  written the way a question would be asked out-retrieves any amount of
  raw material — the same conclusion reached about the cloned repo and
  the binary attachments.

## Considered, not done

- **General credential-name patterns in `/ingest`** (`*credential*`,
  `*secret*`, `*password*`, `*api_key*`). Two specific files are skipped
  by name today; a pattern would also catch future ones. Not added yet —
  it would silently skip legitimate documents *about* authentication,
  which is a real category in data-engineering notes.

## Small

- `mypy` is not in the dev dependency group, so `uv run mypy src` fails
  with "Failed to spawn". Only `uv run pre-commit run mypy` works.
- `CLAUDE.md` refers to `requirements.txt`, which does not exist —
  dependencies live in `pyproject.toml` plus `uv.lock`.
- `llama3.2:3b` is no longer used by anything. `ollama rm llama3.2:3b`
  reclaims 2.02GB.
