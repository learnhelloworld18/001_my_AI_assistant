# Architecture Diagram

Full request-flow diagram for the personal local AI assistant. See
`REQUIREMENTS.md` for the fuller rationale behind each component.

```mermaid
flowchart TD
    User(["User input"]) --> MetaCheck{"Meta-command?"}

    subgraph MetaCommands ["Meta-commands - bypass supervisor"]
        Help["/help<br/>print help text"]
        Ingest["/ingest path [--collection]<br/>run rag/ingest.py"]
        Stats["/stats<br/>query SQLite, print summary"]
        Clear["/clear<br/>reset in-memory session state"]
    end

    MetaCheck -->|"/help"| Help
    MetaCheck -->|"/ingest path [--collection]"| Ingest
    MetaCheck -->|"/stats"| Stats
    MetaCheck -->|"/clear"| Clear
    MetaCheck -->|"/exit"| SessionEnd
    MetaCheck -->|"anything else"| Supervisor["Supervisor<br/>langgraph-supervisor · llama3.2"]

    subgraph Agents ["Specialist agents - one job each"]
        Coding["coding_agent<br/>qwen2.5-coder"]
        Research["research_agent<br/>qwen2.5"]
        Docs["docs_agent<br/>qwen2.5"]
        General["general_agent<br/>llama3.2"]
    end

    Supervisor -->|"routes, can chain"| Coding
    Supervisor --> Research
    Supervisor --> Docs
    Supervisor --> General

    subgraph CodingTools ["coding_agent tools - validate before write, confirm before state change"]
        Validate["validate_code.py<br/>ruff · sqlfluff · terraform · dbt"]
        ConfirmCheck{"State-changing<br/>action?"}
        Safety["safety.py gate<br/>show diff, ask user to confirm"]
        FS["filesystem / git / shell<br/>(scoped to project root)"]
        Aborted["action aborted,<br/>explain to user"]
    end
    Coding --> Validate --> ConfirmCheck
    ConfirmCheck -->|"read-only"| FS
    ConfirmCheck -->|"state-changing"| Safety
    Safety -->|"confirmed"| FS
    Safety -->|"declined"| Aborted

    subgraph ResearchTools ["research_agent tools"]
        Tavily["Tavily web_search"]
        Visit["visit_webpage"]
    end
    Research --> Tavily
    Research --> Visit

    subgraph DocsTools ["docs_agent tools"]
        SearchNotes["search_notes"]
        SearchResume["search_resume"]
    end
    Docs --> SearchNotes
    Docs --> SearchResume

    Embedding["nomic-embed-text<br/>embed text/query"]
    Ingest --> Embedding
    SearchNotes --> Embedding
    SearchResume --> Embedding
    Summarize --> Embedding

    subgraph VectorDB ["Chroma - local vector DB"]
        TechNotes[("tech_notes")]
        ResumeCol[("resume_interview")]
        MemCol[("conversation_memory")]
    end
    Embedding --> TechNotes
    Embedding --> ResumeCol
    Embedding --> MemCol

    FS --> Reply["Agent response"]
    Aborted --> Reply
    Research --> Reply
    Docs --> Reply
    General --> Reply
    Reply --> Supervisor
    Supervisor -->|"final answer"| Stream(["Streamed to terminal"])

    ProcessExit["Ctrl+C / terminal closed"] -.-> SessionEnd
    SessionEnd["Session end"] --> Summarize["memory.py summarizes session<br/>llama3.2"]
    MemCol -.retrieved next session.-> Supervisor

    subgraph Obs ["Observability"]
        Metrics["metrics.py timing wrapper"]
        SQLite[("SQLite log")]
    end
    Supervisor -.-> Metrics
    Coding -.-> Metrics
    Research -.-> Metrics
    Docs -.-> Metrics
    General -.-> Metrics
    Metrics --> SQLite
    Stats --> SQLite

    Ollama["Ollama - local model server"]
    Ollama -.serves.-> Supervisor
    Ollama -.serves.-> Coding
    Ollama -.serves.-> Research
    Ollama -.serves.-> Docs
    Ollama -.serves.-> General
    Ollama -.serves.-> Summarize
    Ollama -.serves.-> Embedding

    classDef flow fill:#e5e7eb,stroke:#6b7280,color:#1f2937
    classDef meta fill:#fde68a,stroke:#d97706,color:#78350f
    classDef orchestration fill:#ddd6fe,stroke:#7c3aed,color:#4c1d95
    classDef agent fill:#bfdbfe,stroke:#2563eb,color:#1e3a8a
    classDef tool fill:#a7f3d0,stroke:#059669,color:#064e3b
    classDef storage fill:#fbcfe8,stroke:#db2777,color:#831843
    classDef obs fill:#fed7aa,stroke:#ea580c,color:#7c2d12
    classDef serving fill:#a5f3fc,stroke:#0891b2,color:#164e63
    classDef stop fill:#fecaca,stroke:#dc2626,color:#7f1d1d

    class User,MetaCheck,ConfirmCheck,Reply,Stream,SessionEnd,ProcessExit flow
    class Help,Ingest,Stats,Clear meta
    class Supervisor orchestration
    class Coding,Research,Docs,General agent
    class Validate,Safety,FS,Tavily,Visit,SearchNotes,SearchResume tool
    class Embedding,TechNotes,ResumeCol,MemCol storage
    class Metrics,SQLite obs
    class Ollama,Summarize serving
    class Aborted stop
```

**Color key**: **gray** = flow control/decision points, **amber** = meta-commands,
**violet** = orchestration (supervisor), **blue** = specialist agents,
**green** = tools/actions, **pink** = RAG/vector storage, **orange** =
observability, **cyan** = model serving, **red** = the one "stop/declined"
state. Renders natively on GitHub (no extra tooling needed).

**Reading it**: a query either short-circuits as a meta-command or goes
to the supervisor, which routes to (and can chain) one or more of the
four specialist agents; each agent's tools are scoped to its own job.
`coding_agent` **validates generated code before deciding whether it
even needs confirmation** — read-only actions proceed straight through,
state-changing ones stop for explicit user approval, and a decline
aborts cleanly rather than silently doing nothing. Every RAG operation
(ingest, both search tools, session-summary storage) passes through the
same embedding step before touching Chroma — there's no direct
tool-to-vector-DB edge anywhere. Every agent is instrumented by the same
timing wrapper regardless of which one ran; meta-commands each have a
real destination rather than one generic handler.
