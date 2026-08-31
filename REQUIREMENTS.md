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

- Entry point: `main.py`, run from the terminal.
- Conversational REPL, similar in feel to Claude Code — fast, streaming
  responses, iterative back-and-forth.
- Responses should feel near-instant for simple queries (small models,
  warm `keep_alive`, streaming output, capped context window).

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

- Build hand-rolled first (own supervisor loop, own routing), for the
  learning value, before optionally trying LangGraph's prebuilt
  `langgraph-supervisor` package for comparison.
- Each agent has exactly one job — do not combine research and
  answer-writing (or any two responsibilities) in a single agent/prompt.

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

## Tool supply — MCP-first where practical

- Reuse existing MCP CLI project and `langchain-mcp-adapters` to plug
  MCP servers in as LangChain tools rather than hand-writing every tool.
- Filesystem, git, shell execution tools for `coding_agent` via MCP.
- Web search/page-visiting tools for `research_agent` — reusable
  directly from prior GAIA agent build (swap backend model to local
  Ollama).
- RAG tool for `docs_agent`, either hand-written or MCP-wrapped.

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

## Suggested module structure

```
001_my_AI_assistant/
  main.py              # REPL entry point
  supervisor.py         # orchestration graph / routing logic
  agents/
    coding_agent.py
    research_agent.py
    docs_agent.py
    general_agent.py
  tools/                 # hand-written + MCP-wrapped tools
  rag/
    ingest.py            # chunk + embed + store personal notes
    query.py              # retrieve + generate
  state.py               # shared graph state schema
```

## Suggested build order

1. `main.py` — bare REPL loop, no routing, one hardcoded agent, to prove
   the plumbing works end to end.
2. Hand-rolled supervisor + `research_agent` (fastest to stand up —
   reuses existing GAIA-project tools directly).
3. `coding_agent` with MCP-backed file/shell/git tools.
4. RAG pipeline (`ingest.py` + `query.py`) + `docs_agent`.
5. `general_agent` for fast low-stakes chat/drafts.
6. Add small-model classifier fallback to the router for ambiguous
   queries.
7. Optionally: swap hand-rolled supervisor for `langgraph-supervisor` to
   compare against the from-scratch version.
