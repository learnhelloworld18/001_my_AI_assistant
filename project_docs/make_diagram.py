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
PATH = Edge(color="black", fontsize="26", penwidth="2")
STOP = Edge(color="firebrick", style="bold", fontsize="26")
DATA = Edge(color="royalblue", fontsize="26", penwidth="2")
# constraint=false on anything that is not part of the forward flow. Without
# it, a feedback edge (the answer returning to the REPL, a tool resuming, Ollama
# serving nine nodes) drags its endpoints out of rank and the bands dissolve -
# which is what made the first version read as scattered rather than as a flow.
TRACE = Edge(color="dimgrey", style="dashed", constraint="false", fontsize="26", penwidth="2.5")
NOTE = Edge(color="dimgrey", style="dotted", constraint="false", fontsize="26", penwidth="2.5")
BACK = Edge(color="black", style="dashed", constraint="false", fontsize="26", penwidth="2.5")
SERVE = Edge(color="darkgreen", style="dotted", constraint="false", fontsize="26")

# Nodes inside a cluster stack in the graph's direction, so a chain of four
# becomes a tall narrow column. Passing LR to the cluster lays its own contents
# out horizontally while the graph as a whole still reads top to bottom.
WIDE = {"rankdir": "LR", "margin": "90"}

# Every cluster gets padding. Without it a node's label - which is wider than
# its icon - spills past the cluster border and reads as belonging to nothing.
BOX = {"margin": "90"}

# Doubled from graphviz's defaults (graph 24, node 14, edge 14). At the sizes
# this renders to, the default is unreadable without zooming twice.
# height as well as fontsize. diagrams places the label inside a fixed-height
# node box, so a five- or six-line label at double size overflows upward and
# runs straight through the icon. Taller boxes give the text somewhere to go.
NODE_ATTR = {"fontsize": "28", "height": "3.2", "imagepos": "tc", "labelloc": "b"}
# Kept for completeness, but diagrams overrides it per edge - the size that
# actually applies is the fontsize passed to each Edge below.
EDGE_ATTR = {"fontsize": "26"}

GRAPH_ATTR = {
    "fontsize": "48",
    "bgcolor": "white",
    # Manhattan routing. Tried and rejected in an earlier pass, when feedback
    # edges still set rank and ortho sent them the long way round; with those
    # now unconstrained it produces clean right angles.
    "splines": "ortho",
    # Ranks the graph by the forward flow only, so it reads top to bottom:
    # you -> REPL -> router -> agents -> tools -> gate -> tier -> back to you.
    "newrank": "true",
    # Scaled with the font. Doubling the text doubled the label widths, and
    # 1.1 - which was right at the old size - put them straight through each
    # other. Spacing has to move with type size, not be set once.
    # Pulled back in from 3.2 once cluster margins took over the job of keeping
    # labels apart. 3.2 was preventing overlap by brute force and left the
    # canvas mostly empty.
    "nodesep": "2.2",
    "ranksep": "1.5",
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
        node_attr=NODE_ATTR,
        edge_attr=EDGE_ATTR,
        outformat="png",
    ):
        user = User("you\nany directory · the one you\nlaunch from is the project")

        with Cluster("REPL  ·  main.py", graph_attr=WIDE):
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

        with Cluster("meta-commands  ·  never reach an agent", graph_attr=WIDE):
            c_help = InputOutput("/help")
            c_clear = InputOutput("/clear\nwipes history,\nkeeps session_id")
            c_ingest = InputOutput("/ingest <path>\n[notes|resume]")
            c_remember = InputOutput("/remember <text>\nstored verbatim")
            c_stats = InputOutput("/stats [all|24h|3d]\ndefaults to this session")

        with Cluster("dragged file  ·  dropped.py", graph_attr=WIDE):
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

        with Cluster(
            "coding_agent  ·  read then write  ·  the coder model cannot call tools",
            graph_attr=BOX,
        ):
            cg_read = Python("read node\nqwen2.5:3b + tools\nreturns evidence, no prose")
            cg_write = Python("write node\nqwen2.5-coder:7b-q4_K_M\nno tools bound")
            cg_gate = Decision("gate")
            with Cluster("tools/coding.py", graph_attr=WIDE):
                cg_list = Python("list_project_files")
                cg_readf = Python("read_project_file\n20k char cap")
                cg_pw = Python("propose_write")
                cg_pc = Python("propose_command")

        with Cluster("tools/safety.py  ·  PROJECT_ROOT = cwd captured at launch", graph_attr=WIDE):
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

        with Cluster("docs_agent  ·  qwen2.5:3b  ·  your own documents", graph_attr=BOX):
            dg = Python("docs_agent\nprompt: never fill a gap\nfrom memory")
            dg_notes = Python("search_notes")
            dg_res = Python("search_resume")
            dg_exp = Python("search_experience\none query per CAREER_ROLE\ncoverage, not ranking")
            dg_gate = Decision("gate\ntop_score >= 0.37")

        with Cluster("research_agent  ·  qwen2.5:3b  ·  the open web", graph_attr=BOX):
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

        with Cluster("contracts  ·  what the evidence gate reads", graph_attr=WIDE):
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
            "UNGROUNDED    unverified\ntiers, never percentages\n"
            "streamed back to the terminal"
        )

        with Cluster("storage  ·  ~/.myassistant  ·  embedded, no server", graph_attr=WIDE):
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

        # In a cluster of its own even though it is a single node: loose at the
        # top rank it was laid over the neighbouring cluster's labels, because
        # graphviz reserves space for a node's box but not for its caption.
        with Cluster("model serving", graph_attr=BOX):
            ollama = Server(
                "Ollama\nlocalhost:11434\nserves every model here\n"
                "qwen2.5:3b\nqwen2.5-coder:7b\nqwen2.5vl:3b\nnomic-embed-text"
            )
        web = Internet("Tavily API\nand the open web")
        # Two words per shape. Longer captions were tried and ran into each
        # other: six nodes on one row means each caption owns very little
        # width, and the legend only has to name the shape, not explain it.
        with Cluster(
            "legend  ·  what each shape means",
            graph_attr={"rankdir": "LR", "margin": "90", "nodesep": "2.6"},
        ):
            Python("PROCESS\nsomething that runs")
            Decision("DECISION\na branch")
            InputOutput("IN / OUT\na command or a refusal")
            Document("DATA\na record, not a step")
            Storage("STORE\non disk")
            Server("SERVER\nlong-running")

        with Cluster("observability  ·  optional, degrades to a no-op", graph_attr=WIDE):
            lf = Grafana(
                "Langfuse v2\nCallbackHandler on the graph\nauth_check once, cached\n"
                "every span, plus the gate's confidence score"
            )
            lfdb = Docker("docker compose\nweb + postgres")

        # --- request path -----------------------------------------------------
        user >> Edge(fontsize="26", label="  types a prompt", color="black", penwidth="2.5") >> repl
        repl >> PATH >> route
        repl >> NOTE >> sess
        repl >> NOTE >> errors

        route >> Edge(fontsize="26", label="starts with /") >> c_help
        route >> PATH >> c_clear
        route >> PATH >> c_ingest
        route >> PATH >> c_remember
        route >> PATH >> c_stats
        route >> Edge(fontsize="26", label="a real file path") >> d_parse
        route >> Edge(fontsize="26", label="a question") >> sup

        d_parse >> PATH >> d_deny
        d_deny >> STOP >> d_refused
        d_deny >> PATH >> d_ask
        d_ask >> Edge(fontsize="26", label="image") >> d_img
        d_ask >> Edge(fontsize="26", label="text") >> d_txt
        d_img >> BACK >> repl
        d_txt >> BACK >> repl

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
            >> Edge(
                fontsize="26",
                label="outside root,\nor a credential",
                color="firebrick",
                style="bold",
            )
            >> s_deny
        )
        s_path >> PATH >> s_int
        s_cmd >> Edge(fontsize="26", label="ALLOW  read-only") >> s_act
        s_cmd >> Edge(fontsize="26", label="CONFIRM") >> s_int
        s_cmd >> Edge(fontsize="26", label="DENY", color="firebrick", style="bold") >> s_deny
        s_int >> Edge(fontsize="26", label="yes") >> s_act
        s_int >> Edge(fontsize="26", label="no", color="firebrick") >> s_declined
        (
            s_act
            >> Edge(
                fontsize="26", label="resumes inside the tool", style="dashed", constraint="false"
            )
            >> cg_read
        )
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
            >> Edge(
                fontsize="26",
                label="walk · skip backups,\nnested repos, credentials",
                color="royalblue",
            )
            >> man
        )
        (
            man
            >> Edge(fontsize="26", label="changed files only", color="royalblue")
            >> chunker
            >> DATA
            >> embed
        )
        (
            embed
            >> Edge(fontsize="26", label="delete-then-add\nper source", color="royalblue")
            >> chroma
        )
        c_remember >> DATA >> mem
        (
            shutdown
            >> Edge(fontsize="26", label="session summary", color="royalblue", constraint="false")
            >> mem
        )
        mem >> DATA >> embed
        (
            chroma
            >> Edge(
                fontsize="26",
                label="recalled once,\nnext session",
                color="royalblue",
                style="dashed",
            )
            >> sup
        )

        # --- contracts and out -------------------------------------------------
        # No edges into the contracts cluster, for the reason Ollama has none:
        # it sits at the top rank and the gate sits at the bottom, so the line
        # spanned the whole canvas and reserved an empty column the width of
        # its own label. Contracts are a reference panel, like the legend - who
        # reads them is in the cluster's title instead.
        _ = (obs, state)
        gate >> PATH >> tiers
        # The answer's return to the REPL is not drawn. tiers sits at the last
        # rank and repl at the first, so the edge was routed around the outside
        # of everything and reserved a column the full height of the canvas for
        # two words. Those two words are in the tiers caption now.

        # --- observability and serving ------------------------------------------
        sup >> TRACE >> lf
        # The gate's score reaches Langfuse the same way every span does, so it
        # is named in the Langfuse caption rather than drawn: as its own edge
        # from the last rank to the first it held open an empty block of canvas
        # roughly the size of an agent cluster.
        (
            c_stats
            >> Edge(
                fontsize="26", label="fetch_traces\nsession or window", color="grey", style="dashed"
            )
            >> lf
        )
        lf >> TRACE >> lfdb
        # No edge at all from Ollama. It serves every model in the picture, so
        # the honest drawing is nine dotted lines - noise that crowds out the
        # paths that actually differ. Collapsing it to one line to the
        # supervisor was worse: with constraint=false across a canvas this
        # wide, graphviz drew a line too faint to follow and the label read as
        # floating text. The fact lives in the node's own caption instead.
        _ = (ollama, sup)


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}.png")
