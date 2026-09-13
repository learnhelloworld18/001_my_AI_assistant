"""Generate the detailed architecture diagram.

    uv run python project_docs/make_diagram.py    ->  ./architecture.png

Needs graphviz on the system (`brew install graphviz`) plus the `diagrams` dev
dependency.

A script rather than a drawn image, deliberately. ARCHITECTURE.md's hand-written
diagram had gone stale in three ways before anyone noticed; a generated one
turns a wrong edge into a diff in review, and lives beside the code it
describes.

The output is gitignored, which is what lets it be this detailed - it renders at
full dpi without fighting pre-commit's 500KB large-file limit. The mermaid
version in ARCHITECTURE.md is the one that renders on GitHub.

Two layout facts, learned by trying them:
  concentrate=true    breaks this graph outright ("rebuild_vlists: lead is null
                      for rank 1") and produces no output at all
  splines=ortho       routes lines the long way round; they cross constantly
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.database import Postgresql
from diagrams.onprem.monitoring import Grafana
from diagrams.onprem.network import Internet
from diagrams.programming.flowchart import Decision, Document, InputOutput
from diagrams.programming.language import Python

# Repo root, not project_docs - it is the first thing you see in the folder.
OUT = Path(__file__).resolve().parent.parent / "architecture"

# Edge colour carries meaning, so the picture reads without the legend:
#   black      the normal request path
#   firebrick  refused or declined - the paths that must stay visible
#   royalblue  storage reads and writes
#   grey       observability and internal contracts
#   darkgreen  model serving
PATH = Edge(color="black")
STOP = Edge(color="firebrick", style="bold")
DATA = Edge(color="royalblue")
TRACE = Edge(color="grey", style="dashed")
NOTE = Edge(color="grey", style="dotted")

GRAPH_ATTR = {
    "fontsize": "24",
    "bgcolor": "white",
    "splines": "spline",
    # Wide, because these labels are four lines each - at 0.45 adjacent nodes
    # overlap and the text becomes unreadable where it matters most.
    "nodesep": "1.1",
    "ranksep": "1.3",
    "pad": "0.6",
    "dpi": "110",
}


def build() -> None:
    """Draw the whole thing.

    Every label is a real module, model, constant or collection: if it is in
    the picture it exists in the code, and the numbers are the ones actually
    configured rather than round approximations.
    """
    with Diagram(
        "myassistant  ·  local-first multi-agent assistant  ·  detailed architecture",
        filename=str(OUT),
        show=False,
        # TB, not LR. With this many clustered nodes LR forces a 4400x8100
        # column that is unreadable however much detail it carries.
        direction="TB",
        graph_attr=GRAPH_ATTR,
        outformat="png",
    ):
        user = User("you\nterminal, any directory")

        with Cluster("REPL  ·  main.py"):
            repl = Python(
                "prompt_toolkit\nFileHistory ~/.myassistant/history\n"
                "completes only on '/'\nstreams tokens · subgraphs=True"
            )
            route = Decision("what is this line?\ncommand / file path / question")
            sess = Document(
                "Session\nhistory: list[BaseMessage]\nsession_id groups traces\n"
                "recall runs once per run"
            )
            errors = InputOutput("per-turn try/except\none bad turn never\nkills the loop")
            shutdown = Python(
                "/exit · Ctrl-D · SIGHUP · SIGTERM\n1. flush Langfuse\n"
                "2. summarise, 20s cap\nsecond signal exits at once"
            )

        with Cluster("meta-commands  ·  never reach an agent"):
            c_help = InputOutput("/help")
            c_clear = InputOutput("/clear\nwipes history,\nkeeps session_id")
            c_ingest = InputOutput("/ingest <path>\n[notes|resume]")
            c_remember = InputOutput("/remember <text>\nstored verbatim")
            c_stats = InputOutput("/stats [all|24h|3d]\ndefaults to this session")

        with Cluster("dragged file  ·  dropped.py"):
            d_parse = Decision("shlex unescape\n'/a/my\\ file.png'\nresolves to a real file?")
            d_deny = Decision("denylist\n.env  *.pem  *.key\n~/.ssh  ~/.aws  ~/.gnupg")
            d_ask = Decision("confirm\nresolved path + size\ndefaults to no")
            d_img = Python(
                "read_image.py\ndownscale to 1600px\nqwen2.5vl:3b · keep_alive 2m\n"
                "reply <120 chars = unreadable"
            )
            d_txt = Python("ingest loaders\nmd · txt · pdf · docx")
            d_refused = InputOutput("refused\nno prompt shown")

        sup = Server(
            "supervisor\nlanggraph-supervisor\nqwen2.5:3b · keep_alive 30m\n"
            "parallel_tool_calls=False\noutput_mode=last_message\nInMemorySaver checkpointer"
        )

        with Cluster("coding_agent  ·  read then write  ·  the coder model cannot call tools"):
            cg_read = Python("read node\nqwen2.5:3b + tools\nreturns evidence, no prose")
            cg_write = Python("write node\nqwen2.5-coder:7b-q4_K_M\nno tools bound")
            cg_gate = Decision("gate")
            with Cluster("tools/coding.py"):
                cg_list = Python("list_project_files")
                cg_readf = Python("read_project_file\n20k char cap")
                cg_pw = Python("propose_write")
                cg_pc = Python("propose_command")

        with Cluster("tools/safety.py  ·  PROJECT_ROOT = cwd captured at launch"):
            s_path = Decision("safe_path()\nresolve() BEFORE the check\ncatches ../.. and symlinks")
            s_cmd = Decision(
                "check_command()\nper segment, strictest wins\nunknown = CONFIRM, never ALLOW"
            )
            s_int = Decision("interrupt()\npauses the whole graph\nresumes on the same line")
            s_act = Python("write the file\nrun it · cwd=PROJECT_ROOT\n60s timeout")
            s_deny = InputOutput(
                "DENY\nsudo · rm -r · dd · chmod 777\ncurl|sh · fork bomb · .env\n"
                "never becomes a question"
            )
            s_declined = InputOutput("declined\nand the agent is told")

        with Cluster("docs_agent  ·  qwen2.5:3b  ·  your own documents"):
            dg = Python("docs_agent\nprompt: never fill a gap\nfrom memory")
            dg_notes = Python("search_notes")
            dg_res = Python("search_resume")
            dg_exp = Python("search_experience\none query per CAREER_ROLE\ncoverage, not ranking")
            dg_gate = Decision("gate\ntop_score >= 0.37")

        with Cluster("research_agent  ·  qwen2.5:3b  ·  the open web"):
            rg = Python("research_agent\nprompt: snippets are\nnot evidence")
            rg_search = Python("web_search · Tavily\nmax 5 · kind=search")
            rg_visit = Python(
                "visit_webpage\nBeautifulSoup + markdownify\nlooks_empty(): <400 chars,\n"
                "consent walls, JS shells"
            )
            rg_gate = Decision("gate\nneeds an ok kind=page")

        general = Python(
            "general_agent\nqwen2.5:3b · no tools\nsingle call, no ReAct loop\nalways UNGROUNDED"
        )

        with Cluster("contracts"):
            obs = Document(
                "Observation (frozen)\nok · detail · content\nsource · metrics{kind,…}\n"
                "render() -> [OK] / [TOOL FAILED]"
            )
            state = Document(
                "AssistantState (TypedDict)\nmessages   add_messages\n"
                "observations   operator.add\nconfidence · remaining_steps"
            )

        gate = Decision(
            "evidence gate\ndeterministic · no model call\nany ok Observation\n"
            "whose kind is not 'search'"
        )
        tiers = Document(
            "HIGH          no tag shown\nLOW           thin evidence\n"
            "UNGROUNDED    unverified\ntiers, never percentages"
        )

        with Cluster("storage  ·  ~/.myassistant  ·  embedded, no server"):
            man = Postgresql(
                "manifest.db (SQLite)\n(source, collection) -> hash\nhashes CONTENT, not mtime"
            )
            chunker = Python(
                "RecursiveCharacterTextSplitter\n1000 chars · 150 overlap\nrole tagged from path"
            )
            embed = Python("nomic-embed-text\nOllamaEmbeddings")
            chroma = Storage("Chroma\ntech_notes\nresume_interview\nconversation_memory")
            mem = Python(
                "rag/memory.py\nsummaries, never transcripts\nrecall threshold 0.18\n"
                "(documents use 0.37)"
            )

        ollama = Server(
            "Ollama · localhost:11434\nqwen2.5:3b · qwen2.5-coder:7b\nqwen2.5vl:3b · nomic-embed-text"
        )
        web = Internet("Tavily API\nand the open web")
        with Cluster("observability  ·  optional, degrades to a no-op"):
            lf = Grafana("Langfuse v2\nCallbackHandler on the graph\nauth_check once, cached")
            lfdb = Docker("docker compose\nweb + postgres")

        # --- request path -----------------------------------------------------
        user >> PATH >> repl >> PATH >> route
        repl >> NOTE >> sess
        repl >> NOTE >> errors

        route >> Edge(label="starts with /") >> c_help
        route >> PATH >> c_clear
        route >> PATH >> c_ingest
        route >> PATH >> c_remember
        route >> PATH >> c_stats
        route >> Edge(label="a real file path") >> d_parse
        route >> Edge(label="a question") >> sup

        d_parse >> PATH >> d_deny
        d_deny >> STOP >> d_refused
        d_deny >> PATH >> d_ask
        d_ask >> Edge(label="image") >> d_img
        d_ask >> Edge(label="text") >> d_txt
        d_img >> PATH >> repl
        d_txt >> PATH >> repl

        sup >> PATH >> cg_read
        sup >> PATH >> dg
        sup >> PATH >> rg
        sup >> PATH >> general

        # --- coding, and the fence -------------------------------------------
        cg_read >> PATH >> cg_list
        cg_read >> PATH >> cg_readf
        cg_read >> PATH >> cg_pw
        cg_read >> PATH >> cg_pc
        cg_readf >> PATH >> s_path
        cg_pw >> PATH >> s_path
        cg_pc >> PATH >> s_cmd
        (
            s_path
            >> Edge(label="outside root,\nor a credential", color="firebrick", style="bold")
            >> s_deny
        )
        s_path >> PATH >> s_int
        s_cmd >> Edge(label="ALLOW  read-only") >> s_act
        s_cmd >> Edge(label="CONFIRM") >> s_int
        s_cmd >> Edge(label="DENY", color="firebrick", style="bold") >> s_deny
        s_int >> Edge(label="yes") >> s_act
        s_int >> Edge(label="no", color="firebrick") >> s_declined
        s_act >> Edge(label="resumes inside the tool", style="dashed") >> cg_read
        cg_read >> PATH >> cg_write >> PATH >> cg_gate >> PATH >> gate

        # --- docs -------------------------------------------------------------
        dg >> PATH >> dg_notes >> DATA >> embed
        dg >> PATH >> dg_res >> DATA >> embed
        dg >> PATH >> dg_exp >> DATA >> embed
        dg >> PATH >> dg_gate >> PATH >> gate

        # --- research ---------------------------------------------------------
        rg >> PATH >> rg_search >> DATA >> web
        rg >> PATH >> rg_visit >> DATA >> web
        rg >> PATH >> rg_gate >> PATH >> gate
        general >> PATH >> gate

        # --- storage ----------------------------------------------------------
        (
            c_ingest
            >> Edge(label="walk · skip backups,\nnested repos, credentials", color="royalblue")
            >> man
        )
        man >> Edge(label="changed files only", color="royalblue") >> chunker >> DATA >> embed
        embed >> Edge(label="delete-then-add\nper source", color="royalblue") >> chroma
        c_remember >> DATA >> mem
        shutdown >> Edge(label="session summary", color="royalblue") >> mem
        mem >> DATA >> embed
        (
            chroma
            >> Edge(label="recalled once,\nnext session", color="royalblue", style="dashed")
            >> sup
        )

        # --- contracts and out -------------------------------------------------
        gate >> NOTE >> obs
        gate >> NOTE >> state
        gate >> PATH >> tiers >> PATH >> repl

        # --- observability and serving ------------------------------------------
        sup >> TRACE >> lf
        gate >> Edge(label="confidence score", color="grey", style="dashed") >> lf
        c_stats >> Edge(label="fetch_traces\nsession or window", color="grey", style="dashed") >> lf
        lf >> TRACE >> lfdb
        for node in (sup, cg_read, cg_write, dg, rg, general, embed, d_img, mem):
            ollama >> Edge(color="darkgreen", style="dotted") >> node


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}.png")
