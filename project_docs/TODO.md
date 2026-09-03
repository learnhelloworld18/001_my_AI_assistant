# TODO

Backlog. `implementation_steps.md` records what is done; this records what
isn't. Move items out as they land.

## Done: drag-and-drop and image reading

Built. See `dropped.py`, `tools/read_image.py`, and DESIGN_DECISIONS.

Left open from that work:

- **HEIC images** are not readable. macOS photos default to HEIC and
  decoding needs a separate Pillow plugin. Two such files exist in the
  documents folder.
- **A dragged file is read once, not remembered.** Confirming grants a
  one-time read; there is no session allowlist. Re-dragging is the way to
  re-read. Fine so far; revisit if it becomes annoying.
- **Individual labels from a diagram should not be trusted verbatim.**
  The model reads most of them correctly and mangles some ("Articles/Asse.
  Regulations"). Good enough to understand a diagram, not to quote from.

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
