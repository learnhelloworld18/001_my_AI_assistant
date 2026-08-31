# Architecture Diagram

Full request-flow diagram for the personal local AI assistant. See
`REQUIREMENTS.md` for the fuller rationale behind each component.

```mermaid
flowchart TD
    User(["User input"]) --> MetaCheck{"Meta-command?"}
    MetaCheck -->|"/help /exit /stats /ingest /clear"| MetaHandler["Handled directly by main.py"]
    MetaCheck -->|"no"| Supervisor["Supervisor<br/>langgraph-supervisor · llama3.2"]

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

    subgraph CodingTools ["coding_agent tools"]
        Safety["safety.py gate<br/>scope + confirm + denylist"]
        FS["filesystem / git / shell"]
        Validate["validate_code.py<br/>ruff · sqlfluff · terraform · dbt"]
    end
    Coding --> Safety --> FS
    Coding --> Validate

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

    subgraph VectorDB ["Chroma - local vector DB"]
        TechNotes[("tech_notes")]
        ResumeCol[("resume_interview")]
        MemCol[("conversation_memory")]
    end
    SearchNotes --> TechNotes
    SearchResume --> ResumeCol

    Coding --> Reply["Agent response"]
    Research --> Reply
    Docs --> Reply
    General --> Reply
    Reply --> Supervisor
    Supervisor -->|"final answer"| Stream(["Streamed to terminal"])

    ExitEvent["/exit or session end"] --> Summarize["memory.py summarizes session<br/>llama3.2"]
    Summarize --> MemCol
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

    Ollama["Ollama - local model server"]
    Ollama -.serves.-> Supervisor
    Ollama -.serves.-> Coding
    Ollama -.serves.-> Research
    Ollama -.serves.-> Docs
    Ollama -.serves.-> General
```

Renders natively on GitHub (no extra tooling needed). Read top to
bottom: a query either short-circuits as a meta-command, or goes to the
supervisor, which routes to (and can chain) one or more of the four
specialist agents; each agent's tools are scoped to its own job;
`docs_agent`'s tools and cross-session memory both live in the same
local Chroma store, just different collections; every agent is
instrumented by the same timing wrapper regardless of which one ran.
