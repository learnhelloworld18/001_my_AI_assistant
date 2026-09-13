"""Generate the architecture diagram. Run: uv run python project_docs/make_diagram.py

Kept as a script rather than a drawn image so the diagram cannot quietly go
stale the way ARCHITECTURE.md's did - it is regenerated from a file that lives
next to the code it describes, and a wrong edge here is a diff in review.

Needs graphviz on the system (brew install graphviz) plus the `diagrams`
dev dependency.
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.database import Postgresql
from diagrams.onprem.monitoring import Grafana
from diagrams.onprem.network import Internet
from diagrams.programming.flowchart import Decision, Document, InputOutput
from diagrams.programming.language import Python

OUT = Path(__file__).parent / "architecture"

# Edge colours carry meaning, so the picture is readable without the legend:
#   black  the normal request path
#   red    something refused or declined
#   blue   storage reads and writes
#   grey   observability, which touches everything and explains nothing
PATH = Edge(color="black")
STOP = Edge(color="firebrick", style="bold")
DATA = Edge(color="royalblue")
TRACE = Edge(color="grey", style="dashed")

GRAPH_ATTR = {
    "fontsize": "22",
    "bgcolor": "white",
    # curved rather than orthogonal: with this many cross-cluster edges, ortho
    # routing sends lines the long way round and they cross constantly.
    "splines": "spline",
    "nodesep": "0.4",
    "ranksep": "1.1",
    "pad": "0.5",
    # Keeps the PNG under pre-commit's 500KB large-file limit. At the default
    # dpi the same graph renders at 533KB and the commit is rejected.
    "dpi": "70",
    # concentrate=true was tried and breaks this graph outright:
    # "rebuild_vlists: lead is null for rank 1", no output at all.
}


def build() -> None:
    """Draw it. Every label is a real module, model or collection name.

    Two things are deliberately *not* drawn. Ollama serves all eight models, and
    Langfuse traces every node - edges for either would be a dozen lines across
    the whole canvas saying "yes, everything". Both are shown as one node with a
    caption instead. A diagram whose edges all mean "everything connects to
    this" has stopped being a diagram.
    """
    with Diagram(
        "myassistant - local-first multi-agent assistant",
        filename=str(OUT),
        show=False,
        direction="LR",
        graph_attr=GRAPH_ATTR,
        # PNG only. The SVG that `diagrams` emits references its 29 icons by
        # absolute path inside .venv, so it renders as broken images anywhere
        # but the machine that made it - useless for a repo.
        outformat="png",
    ):
        user = User("you")

        with Cluster("REPL  ·  main.py"):
            repl = Python("prompt_toolkit\nstreams tokens as they arrive")
            route = Decision("command?\nfile path?\nquestion?")
            cmds = InputOutput(
                "/help  /clear  /ingest\n/remember  /stats  /exit\nnever reach an agent"
            )
            shutdown = Python("/exit · Ctrl-D · SIGHUP\nflush, then summarise")

        with Cluster("dragged file  ·  dropped.py"):
            drop_deny = Decision("denylist\n.env  *.pem  ~/.ssh")
            drop_ask = Decision("confirm\nresolved path + size")
            drop_read = Python("read_image.py  qwen2.5vl\nor the ingest loaders")

        sup = Server("supervisor\nlanggraph-supervisor\nqwen2.5:3b\nroutes only, never answers")

        with Cluster("coding_agent  ·  two models, one job each"):
            c_read = Python("read\nqwen2.5:3b + tools\nproduces no prose")
            c_write = Python("write\nqwen2.5-coder:7b\nno tools bound")
            with Cluster("safety.py  ·  every action passes through"):
                verdict = Decision("ALLOW · CONFIRM · DENY")
                interrupt = Decision("interrupt()\npauses the whole graph")
                act = Python("write the file\nrun the command")
                denied = InputOutput("refused outright\nnever becomes a question")

        with Cluster("docs_agent  ·  qwen2.5:3b"):
            docs = Python("docs_agent")
            t_rag = Python("search_notes\nsearch_resume\nsearch_experience")

        with Cluster("research_agent  ·  qwen2.5:3b"):
            research = Python("research_agent")
            t_web = Python("web_search · Tavily\nvisit_webpage")

        general = Python("general_agent\nqwen2.5:3b\nno tools · always UNGROUNDED")

        with Cluster("storage  ·  ~/.myassistant  ·  no server"):
            man = Postgresql("manifest.db\nsource to content hash\nunchanged files cost nothing")
            embed = Python("nomic-embed-text")
            chroma = Storage("Chroma\ntech_notes\nresume_interview\nconversation_memory")

        gate = Decision("evidence gate\ndeterministic, no model call\nreads Observation.ok")
        tiers = Document("HIGH  no tag shown\nLOW  thin evidence\nUNGROUNDED  unverified")

        ollama = Server("Ollama\nserves every model\nlocal, no API key")
        lf = Grafana("Langfuse  ·  optional\ntraces every node\n/stats reads it back")
        web = Internet("the web")

        # --- the request path -------------------------------------------------
        user >> PATH >> repl >> PATH >> route
        route >> Edge(label="command") >> cmds
        route >> Edge(label="file path") >> drop_deny
        route >> Edge(label="question") >> sup

        drop_deny >> STOP >> InputOutput("refused")
        drop_deny >> PATH >> drop_ask >> PATH >> drop_read

        sup >> PATH >> c_read
        sup >> PATH >> docs
        sup >> PATH >> research
        sup >> PATH >> general

        # --- coding_agent, and the fence --------------------------------------
        c_read >> PATH >> verdict
        verdict >> Edge(label="read-only") >> act
        verdict >> Edge(label="changes state") >> interrupt
        verdict >> Edge(label="sudo · rm -rf · .env", color="firebrick", style="bold") >> denied
        interrupt >> Edge(label="yes") >> act
        interrupt >> Edge(label="no", color="firebrick") >> denied
        act >> Edge(label="resumes inside the tool", style="dashed") >> c_read
        c_read >> PATH >> c_write

        # --- tools and storage ------------------------------------------------
        research >> PATH >> t_web >> DATA >> web
        docs >> PATH >> t_rag >> DATA >> embed
        cmds >> Edge(label="/ingest", color="royalblue") >> man >> DATA >> embed
        embed >> DATA >> chroma
        shutdown >> Edge(label="session summary", color="royalblue") >> chroma
        chroma >> Edge(label="recalled next session", color="royalblue", style="dashed") >> sup

        # --- the gate, and back out -------------------------------------------
        c_write >> PATH >> gate
        docs >> PATH >> gate
        research >> PATH >> gate
        general >> PATH >> gate
        gate >> PATH >> tiers >> PATH >> repl
        drop_read >> PATH >> repl

        # One edge each, not a fan-out - see the docstring.
        gate >> Edge(label="confidence score", color="grey", style="dashed") >> lf
        sup >> Edge(color="darkgreen", style="dotted") >> ollama


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}.png")
