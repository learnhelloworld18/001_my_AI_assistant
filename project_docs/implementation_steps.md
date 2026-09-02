1. project planning
2. repo scaffolding
3. Choose ollama models - coding , research, general llm etc
4. Docker desktop
5. Run Langfuse on docker

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
