1. project planning
2. repo scaffolding
3. Choose ollama models - coding , research, general llm etc

 ┌──────────────────────────────────┬─────────┬─────────────────────────────────────────────────────────────────────────┐
 │              Model               │  Size   │                                Used for                                 │
 ├──────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
 │ qwen2.5:3b-instruct              │ 1.93 GB │ supervisor (routing), research_agent, docs_agent, general_agent, critic │
 ├──────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
 │ qwen2.5-coder:7b-instruct-q4_K_M │ 4.68 GB │ coding_agent — not built yet (step 3)                                   │
 ├──────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
 │ nomic-embed-text                 │ 0.27 GB │ RAG embeddings — ingest and query                                       │
 ├──────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
 ├ qwen2.5vl 3 GB                  │ 3 GB  │       Visual document understanding                                        │
 └──────────────────────────────────┴─────────┴─────────────────────────────────────────────────────────────────────────┘
note - tested granite , moondream - not accurate - eg granite gave inconsistent results - given the context of the image in the project ,
then the model would be able to pick it otherwise if we just drag and drop the image with a generic promp, model does a bad job
Techniques to improve - downscale image,
1. Docker desktop
2. Run Langfuse on docker

## Langfuse on Docker — everyday commands

Run from the repo root:

```bash
cd /Users/him/learn-C-One/001_my_AI_assistant

docker compose -f docker-compose.langfuse.yml up -d          # start (first run pulls images)
docker compose -f docker-compose.langfuse.yml ps             # what's running
docker compose -f docker-compose.langfuse.yml logs -f        # follow logs
docker compose -f docker-compose.langfuse.yml restart        # restart both
docker compose -f docker-compose.langfuse.yml down           # stop, KEEP data
docker compose -f docker-compose.langfuse.yml down -v        # stop, DELETE data
```
6. Get Tavily credentials online - Free tier
7. Get Langfuse secret key and public key
        1. Open http://localhost:3000
        2. Sign up — local account, any email, no verification
        3. Create an Organization, then a Project
        4. Project → Settings → API Keys → Create new API key
8.  Write config.py
9.  Write main.py
10. Created langfuse_client.py
11. Added state.py
12. Now the Tools -- websearch.py , visit_webpage.py
13. Agent - research_ageny.py
14. Added general_agent.py -- before writing the supervisor, we need atleast 2 agents to test routing. It is the cheapest agent we have: a model, a prompt, no tools, no new dependencies, and its confidence tier is a constant (UNGROUNDED). The other two agents - coding_agent and docs_agent are blocked by other dependencies like tools/safety.py and RAG pipeline (chromadb and langchain-chroma), respectively
15. Added supervisor.py
16. Added streaming (continuous streaming output in the terminal when application is processing instructions)
17. Fix the supervisor talking twice - Root-level updates are the right signal; they fire exactly on subgraph completion
18. RAG implementation - first chroma db , lanchain-chroma, and  pypdf + python-docx
    Chroma - no server needed. Its just a python API that break your documents into chunks, run each chunk through an embedding model to turn it into a vector (a list of numbers capturing its meaning), and store those vectors somewhere searchable.
    NOTE - it works well with ttxt data and not ansk
19. Wrote rag/store.py, rag/manifest.py,
20. Add ingest.py
    collections used -
        ├── tech_notes            general technical reference — Spark, Kafka, cloud docs
        ├── resume_interview      your CV, STAR answers, interview prep, work history
        └── conversation_memory   session summaries (not built yet)
21. add query.py
22. Add Tools/search_notes.py
23. Add agents/docs_agent.py

you ask "what did I do at Capital One?"
   ↓
supervisor
   ↓
docs_agent          ← agents/docs_agent.py   THE WORKER
   │                  a model + a prompt + a loop + a confidence gate
   │
   │  decides which tool to call
   ↓
search_resume       ← tools/search_notes.py  THE CAPABILITY
search_experience      three @tool functions. No model, no loop.
search_notes           Each just calls rag.query and returns an Observation.
   ↓
Chroma

24. Visual model - qwen - to read images and screenshots
25. safety.py - The boundary around coding_agent
26. coding agent (note - this is an agent and not a model -- used for understanding code and not for generating code, which a code
    generating LLM does for eg.)
27. MCP doc sources
28. critic.py
29. coding agent.py
30. rag/memory.py, then wiring it into the REPL — /remember
31. observability/stats.py
32. coding agent.py -- graph interrupt() --  so the REPL can ask before acting (restrictions on some shell commands etc)
         it propagates from a tool, inside an agent subgraph, through the supervisor
         Interrupt and resume both work at the agent level — but it doesn't propagate through the supervisor.
         The issue happens because interrupt() is scoped to the specific LangGraph subgraph in which it is raised.When you use langgraph-supervisor to manage agent handoffs, the supervisor orchestrates multiple agents as separate graph nodes or subgraphs. If a tool inside the coding_agent triggers an interrupt(), that interrupt pauses the coding_agent's local graph. However, the supervisor agent treats this handoff or node transition dynamically. If the supervisor's state machine doesn't explicitly bubble up or bubble down the interrupt state across the handoff boundary, the parent graph continues executing its loop, effectively swallowing the pause before it reaches your top-level REPL.

         solution --  a checkpointer on both graphs.
                # agents/coding_agent.py
                graph.compile(name=NAME, checkpointer=InMemorySaver())

                # supervisor.py
                create_supervisor(...).compile(checkpointer=InMemorySaver())
         a checkpointer is how LangGraph saves the paused state. Without one on the agent, there's nothing to pause into. Without one on the supervisor, Command(resume=True) has no thread to resume — it raises "Cannot use Command(resume=...) without checkpointer". They don't need to be the same instance; both just need one

33. /stats implementation
