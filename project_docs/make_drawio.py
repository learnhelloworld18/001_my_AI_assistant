"""Generate the architecture diagram as an editable draw.io file.

    uv run python project_docs/make_drawio.py

writes two files at the repo root:

    architecture.drawio          import this into Lucidchart or draw.io
    architecture_layout.png      a preview, so the layout can be checked
                                 without opening either tool

Why this exists alongside make_diagram.py
-----------------------------------------
make_diagram.py renders a PNG through graphviz. That output is fixed - you
cannot move a box in it - and it has a structural flaw this file does not:
the `diagrams` library draws a node as an *icon* with its caption rendered
OUTSIDE the box (labelloc=b). Graphviz reserves space for the icon and knows
nothing about the width of the text hanging under it, so a four-line caption is
zero pixels wide as far as the layout engine is concerned. Every overlap in
that PNG traces back to this, which is why it was only ever fixable by nudging
margins and nodesep.

Here the text lives INSIDE the box and the box width is set by hand, so
overlap is not something to tune - it cannot be expressed. `assert_no_overlap`
checks it anyway, because a layout constant is easy to mistype.

Making the text bigger
----------------------
Scaling the type and the boxes together does nothing: fit the page to a screen
and everything grows in proportion, so the text reads exactly as small as
before. What makes type read bigger is growing the boxes while holding the
GAPS between them constant - the text then occupies more of the canvas.

That is why positions are computed rather than written down. BOX scales the
type and the boxes; GAP_* are fixed pixel constants that BOX does not touch.
Raising BOX tightens the whole diagram around its own text.

Layout
------
Hand-placed in bands rather than solved by an engine, because the flow it has
to show is already known and an engine has to be argued with:

    you -> REPL -> what is this line? -> three branches
    the question branch continues: supervisor -> agents -> tools -> gate -> tier

Reference panels (contracts, legend, model serving, observability) sit in a
strip at the bottom. They are not steps, and letting them join the flow is what
dragged long edges across the graphviz version.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT_XML = ROOT / "architecture.drawio"
OUT_PNG = ROOT / "architecture_layout.png"

# How much bigger than draw.io's 12px default everything to do with a box is:
# the type, the padding, the line height, the widths passed to node(). The gaps
# below are deliberately NOT multiplied by it - see the module docstring.
BOX = 1.7

FONT_SIZE = round(12 * BOX)
LINE_H = round(16 * BOX)
PAD_X, PAD_Y = round(12 * BOX), round(10 * BOX)
# 0.58 * size is a deliberately pessimistic average glyph width for Helvetica.
# Overestimating wraps a line early, which is harmless; underestimating pushes
# text out of its box, which is the one thing this file exists to prevent.
CHAR_W = FONT_SIZE * 0.58

# Fixed gaps. Constant in pixels whatever BOX is set to.
GAP_X = 46  # between boxes side by side
GAP_Y = 34  # between boxes stacked inside a cluster
GAP_C = 70  # between clusters, horizontally
GAP_B = 84  # between bands, vertically
MARGIN = 40

# A rhombus only offers its text the middle band of its bounding box, so a
# decision needs a bigger box than its caption alone would suggest.
DECISION_SLACK_W, DECISION_SLACK_H = 1.25, 1.7

# Fill per kind. Colour repeats the shape rather than replacing it, so the
# diagram survives being printed in grey.
KINDS = {
    "process": ("rounded=1;arcSize=12;", "#e8eef9", "#3b5f9e"),
    "decision": ("rhombus;", "#fdf3d8", "#b5892a"),
    "inout": (
        "shape=parallelogram;perimeter=parallelogramPerimeter;fixedSize=1;",
        "#eae6f7",
        "#6b52ae",
    ),
    "data": ("shape=document;", "#e6f4ea", "#3a7d4f"),
    "store": ("shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=14;", "#fdeae6", "#b4553a"),
    "server": ("shape=hexagon;perimeter=hexagonPerimeter2;", "#e2f0f4", "#2f7b8c"),
    "actor": ("shape=actor;", "#ffffff", "#333333"),
    "cloud": ("ellipse;shape=cloud;", "#f2f2f2", "#666666"),
}

# Edge styles. endArrow is set explicitly rather than left to the default, so
# direction survives a round trip through an importer that has its own idea of
# what an unstyled edge looks like.
EDGES = {
    "path": "strokeColor=#333333;strokeWidth=2;endArrow=classic;endFill=1;",
    "stop": "strokeColor=#b02c2c;strokeWidth=2;endArrow=classic;endFill=1;",
    "data": "strokeColor=#2f5fbf;strokeWidth=2;endArrow=classic;endFill=1;",
    "back": "strokeColor=#333333;strokeWidth=2;dashed=1;endArrow=classic;endFill=1;",
    "trace": (
        "strokeColor=#777777;strokeWidth=2;dashed=1;dashPattern=1 4;endArrow=open;endFill=0;"
    ),
}
ARROW_COLOURS = {"stop": "#b02c2c", "data": "#2f5fbf", "trace": "#999999"}


def _wrap(label: str, width_px: float) -> list[str]:
    """Break a caption to fit a box, honouring the author's own line breaks."""
    cols = max(8, int((width_px - 2 * PAD_X) / CHAR_W))
    lines: list[str] = []
    for para in label.split("\n"):
        lines.extend(textwrap.wrap(para, cols) or [""])
    return lines


@dataclass
class Node:
    id: str
    label: str
    kind: str
    x: float
    y: float
    w: float
    h: float = 0.0
    lines: list[str] = field(default_factory=list)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.right, self.bottom)


@dataclass
class Cluster:
    id: str
    title: str
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class EdgeSpec:
    src: str
    dst: str
    label: str
    style: str


class Layout:
    """Nodes at computed coordinates, clusters sized from what they contain."""

    TITLE_H = round(30 * BOX)
    CPAD = 20

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.clusters: dict[str, Cluster] = {}
        self.edges: list[EdgeSpec] = []

    def node(self, nid: str, label: str, kind: str, x: float, y: float, w: float = 250) -> Node:
        """Place one box. w is a design width; BOX scales it with the type."""
        w *= BOX
        lines = _wrap(label, w)
        h = max(52.0 * BOX, 2 * PAD_Y + len(lines) * LINE_H)
        if kind == "decision":
            # Widen first, then re-wrap: a rhombus wastes its corners, so the
            # caption needs the extra width before its height is decided.
            w *= DECISION_SLACK_W
            lines = _wrap(label, w)
            h = max(70.0 * BOX, (2 * PAD_Y + len(lines) * LINE_H) * DECISION_SLACK_H)
        n = Node(nid, label, kind, x, y, w, h, lines)
        self.nodes[nid] = n
        return n

    def cluster(self, cid: str, title: str, members: list[str]) -> Cluster:
        """Draw a titled backdrop around members, sized to fit them."""
        rects = [self.nodes[m].rect for m in members]
        x0 = min(r[0] for r in rects) - self.CPAD
        y0 = min(r[1] for r in rects) - self.CPAD - self.TITLE_H
        x1 = max(r[2] for r in rects) + self.CPAD
        y1 = max(r[3] for r in rects) + self.CPAD
        c = Cluster(cid, title, x0, y0, x1 - x0, y1 - y0)
        self.clusters[cid] = c
        return c

    def edge(self, src: str, dst: str, label: str, style: str = "path") -> None:
        self.edges.append(EdgeSpec(src, dst, label, style))

    def shift(self, node_ids: list[str], cluster_ids: list[str], dx: float) -> None:
        """Slide a band sideways once the full canvas width is known."""
        for nid in node_ids:
            self.nodes[nid].x += dx
        for cid in cluster_ids:
            self.clusters[cid].x += dx

    # -- checks ----------------------------------------------------------
    def assert_no_overlap(self, gap: float = 6.0) -> None:
        """No two boxes may touch. This is the whole point of the file.

        Cheap O(n^2) over ~60 nodes. It catches a mistyped coordinate at
        generation time instead of after the import into Lucidchart.
        """
        items = list(self.nodes.values())
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                ax0, ay0, ax1, ay1 = a.rect
                bx0, by0, bx1, by1 = b.rect
                if ax0 < bx1 + gap and bx0 < ax1 + gap and ay0 < by1 + gap and by0 < ay1 + gap:
                    raise AssertionError(f"{a.id} and {b.id} overlap: {a.rect} vs {b.rect}")

    def assert_clusters_clean(self) -> None:
        """A backdrop may only contain its own members.

        Two clusters may overlap only when one nests inside the other
        (tools/coding.py sits inside coding_agent). Anything else means a
        band grew into its neighbour, which reads as a box belonging to the
        wrong group - the same failure as overlapping text, one level up.
        """
        members: dict[str, set[str]] = {}
        for cid, c in self.clusters.items():
            members[cid] = {
                n.id
                for n in self.nodes.values()
                if c.x <= n.x and n.right <= c.right and c.y <= n.y and n.bottom <= c.bottom
            }
        for cid, c in self.clusters.items():
            for oid, o in self.clusters.items():
                if cid >= oid:
                    continue
                inside = c.x >= o.x and c.right <= o.right and c.y >= o.y and c.bottom <= o.bottom
                outside = o.x >= c.x and o.right <= c.right and o.y >= c.y and o.bottom <= c.bottom
                hits = c.x < o.right and o.x < c.right and c.y < o.bottom and o.y < c.bottom
                if hits and not (inside or outside):
                    raise AssertionError(f"clusters {cid} and {oid} overlap without nesting")

    def assert_every_edge_labelled(self) -> None:
        """An unlabelled line makes the reader guess what the arrow means."""
        bare = [f"{e.src}->{e.dst}" for e in self.edges if not e.label.strip()]
        if bare:
            raise AssertionError(f"unlabelled edges: {bare}")

    def bounds(self) -> tuple[float, float]:
        xs = [c.right for c in self.clusters.values()] + [n.right for n in self.nodes.values()]
        ys = [c.bottom for c in self.clusters.values()] + [n.bottom for n in self.nodes.values()]
        return max(xs) + MARGIN, max(ys) + MARGIN


def compose() -> Layout:
    """The diagram itself. Every label is a real module, model or constant."""
    L = Layout()

    # --- band 0/1: you, and the REPL --------------------------------------
    # Placed at x=0 and slid into place at the end, once the widest band below
    # has decided how wide the canvas actually is.
    user = L.node(
        "user", "you\nany directory · the one you\nlaunch from is the project", "actor", 0, 0, 200
    )

    yr = user.bottom + GAP_B
    repl = L.node(
        "repl",
        "prompt_toolkit\nFileHistory ~/.myassistant/history\n"
        "completes only on '/'\nstreams tokens · subgraphs=True",
        "process",
        0,
        yr,
        270,
    )
    L.node(
        "shutdown",
        "/exit · Ctrl-D · SIGHUP · SIGTERM\n1. flush Langfuse\n"
        "2. summarise, 20s cap\nsecond signal exits at once",
        "process",
        repl.right + GAP_X,
        yr,
        270,
    )
    # Under prompt_toolkit rather than beside it. Neither of these is a peer of
    # the REPL in the flow - the Session is what it appends to and the
    # try/except is what it runs inside - so a row below reads as "belongs to"
    # where a row beside reads as "comes after". It also lets the two edges
    # arrive from above, which is the only direction the preview renderer
    # draws an arrowhead for correctly.
    sess = L.node(
        "sess",
        "Session\nhistory: list[BaseMessage]\nsession_id groups traces\nrecall runs once per run",
        "data",
        0,
        repl.bottom + GAP_Y,
        250,
    )
    L.node(
        "errors",
        "per-turn try/except\none bad turn never\nkills the loop",
        "inout",
        sess.right + GAP_X,
        sess.y,
        230,
    )
    c_repl = L.cluster("c_repl", "REPL  ·  main.py", ["repl", "sess", "errors", "shutdown"])
    user.x = c_repl.x + (c_repl.w - user.w) / 2

    route = L.node(
        "route",
        "what is this line?\ncommand / file path / question",
        "decision",
        0,
        c_repl.bottom + GAP_B,
        240,
    )
    route.x = c_repl.x + (c_repl.w - route.w) / 2
    top_nodes = ["user", "repl", "sess", "errors", "shutdown", "route"]

    # --- band 2: the three branches ---------------------------------------
    yb = route.bottom + GAP_B

    d_parse = L.node(
        "d_parse",
        "shlex unescape\n'/a/my\\ file.png'\nresolves to a real file?",
        "decision",
        MARGIN,
        yb,
        210,
    )
    d_deny = L.node(
        "d_deny",
        "denylist\n.env  *.pem  *.key\n~/.ssh  ~/.aws  ~/.gnupg",
        "decision",
        MARGIN,
        d_parse.bottom + GAP_Y,
        210,
    )
    L.node("d_refused", "refused\nno prompt shown", "inout", d_deny.right + GAP_X, d_deny.y, 200)
    d_ask = L.node(
        "d_ask",
        "confirm\nresolved path + size\ndefaults to no",
        "decision",
        MARGIN,
        d_deny.bottom + GAP_Y,
        210,
    )
    d_img = L.node(
        "d_img",
        "read_image.py\ndownscale to 1600px\nqwen2.5vl:3b · keep_alive 2m\n"
        "reply <120 chars = unreadable",
        "process",
        MARGIN,
        d_ask.bottom + GAP_Y,
        250,
    )
    L.node(
        "d_txt",
        "ingest loaders\nmd · txt · pdf · docx",
        "process",
        d_img.right + GAP_X,
        d_img.y,
        220,
    )
    c_drop = L.cluster(
        "c_drop",
        "dragged file  ·  dropped.py",
        ["d_parse", "d_deny", "d_refused", "d_ask", "d_img", "d_txt"],
    )

    xm = c_drop.right + GAP_C
    m1 = L.node("c_help", "/help", "inout", xm, yb, 230)
    m2 = L.node(
        "c_clear", "/clear\nwipes history, keeps session_id", "inout", xm, m1.bottom + GAP_Y, 230
    )
    m3 = L.node("c_ingest", "/ingest <path>\n[notes|resume]", "inout", xm, m2.bottom + GAP_Y, 230)
    m4 = L.node(
        "c_remember", "/remember <text>\nstored verbatim", "inout", xm, m3.bottom + GAP_Y, 230
    )
    L.node(
        "c_stats",
        "/stats [all|24h|3d]\ndefaults to this session",
        "inout",
        xm,
        m4.bottom + GAP_Y,
        230,
    )
    c_meta = L.cluster(
        "c_meta",
        "meta-commands  ·  never reach an agent",
        ["c_help", "c_clear", "c_ingest", "c_remember", "c_stats"],
    )

    sup = L.node(
        "sup",
        "supervisor\nlanggraph-supervisor\nqwen2.5:3b · keep_alive 30m\n"
        "parallel_tool_calls=False\noutput_mode=last_message\nInMemorySaver checkpointer",
        "server",
        c_meta.right + GAP_C,
        yb,
        280,
    )

    xs = sup.right + GAP_C
    man = L.node(
        "man",
        "manifest.db (SQLite)\n(source, collection) -> hash\nhashes CONTENT, not mtime",
        "store",
        xs,
        yb,
        250,
    )
    chunker = L.node(
        "chunker",
        "RecursiveCharacterTextSplitter\n1000 chars · 150 overlap\nrole tagged from path",
        "process",
        man.right + GAP_X,
        yb,
        250,
    )
    mem = L.node(
        "mem",
        "rag/memory.py\nsummaries, never transcripts\nrecall threshold 0.18\n(documents use 0.37)",
        "process",
        xs,
        man.bottom + GAP_Y,
        250,
    )
    L.node("embed", "nomic-embed-text\nOllamaEmbeddings", "process", chunker.x, mem.y, 250)
    L.node(
        "chroma",
        "Chroma\ntech_notes\nresume_interview\nconversation_memory",
        "store",
        xs,
        mem.bottom + GAP_Y,
        250,
    )
    c_store = L.cluster(
        "c_store",
        "storage  ·  ~/.myassistant  ·  embedded, no server",
        ["man", "chunker", "mem", "embed", "chroma"],
    )

    # --- band 3: the agents, one job each ----------------------------------
    ya = max(c_drop.bottom, c_meta.bottom, sup.bottom, c_store.bottom) + GAP_B

    cg_read = L.node(
        "cg_read",
        "read node\nqwen2.5:3b + tools\nreturns evidence, no prose",
        "process",
        MARGIN,
        ya,
        230,
    )
    cg_list = L.node(
        "cg_list", "list_project_files", "process", MARGIN, cg_read.bottom + GAP_Y, 210
    )
    L.node(
        "cg_readf",
        "read_project_file\n20k char cap",
        "process",
        cg_list.right + GAP_X,
        cg_list.y,
        210,
    )
    cg_pw = L.node("cg_pw", "propose_write", "process", MARGIN, cg_list.bottom + GAP_Y, 210)
    L.node("cg_pc", "propose_command", "process", cg_pw.right + GAP_X, cg_pw.y, 210)
    c_tools = L.cluster("c_tools", "tools/coding.py", ["cg_list", "cg_readf", "cg_pw", "cg_pc"])
    cg_write = L.node(
        "cg_write",
        "write node\nqwen2.5-coder:7b-q4_K_M\nno tools bound",
        "process",
        MARGIN,
        c_tools.bottom + GAP_Y,
        230,
    )
    L.node("cg_gate", "gate", "decision", cg_write.right + GAP_X, cg_write.y, 130)
    c_code = L.cluster(
        "c_code",
        "coding_agent  ·  read then write  ·  the coder cannot call tools",
        ["cg_read", "cg_list", "cg_readf", "cg_pw", "cg_pc", "cg_write", "cg_gate"],
    )

    xd = c_code.right + GAP_C
    dg = L.node("dg", "docs_agent\nprompt: never fill a gap\nfrom memory", "process", xd, ya, 230)
    dg_notes = L.node("dg_notes", "search_notes", "process", xd, dg.bottom + GAP_Y, 200)
    L.node("dg_res", "search_resume", "process", dg_notes.right + GAP_X, dg_notes.y, 200)
    dg_exp = L.node(
        "dg_exp",
        "search_experience\none query per CAREER_ROLE\ncoverage, not ranking",
        "process",
        xd,
        dg_notes.bottom + GAP_Y,
        230,
    )
    L.node("dg_gate", "gate\ntop_score >= 0.37", "decision", dg_exp.right + GAP_X, dg_exp.y, 150)
    c_docs = L.cluster(
        "c_docs",
        "docs_agent  ·  qwen2.5:3b  ·  your own documents",
        ["dg", "dg_notes", "dg_res", "dg_exp", "dg_gate"],
    )

    xrg = c_docs.right + GAP_C
    rg = L.node("rg", "research_agent\nprompt: snippets are\nnot evidence", "process", xrg, ya, 230)
    rg_search = L.node(
        "rg_search",
        "web_search · Tavily\nmax 5 · kind=search",
        "process",
        xrg,
        rg.bottom + GAP_Y,
        210,
    )
    L.node(
        "rg_visit",
        "visit_webpage\nBeautifulSoup + markdownify\nlooks_empty(): <400 chars,\n"
        "consent walls, JS shells",
        "process",
        rg_search.right + GAP_X,
        rg_search.y,
        230,
    )
    L.node("rg_gate", "gate\nneeds an ok kind=page", "decision", xrg, rg_search.bottom + GAP_Y, 160)
    c_res = L.cluster(
        "c_res",
        "research_agent  ·  qwen2.5:3b  ·  the open web",
        ["rg", "rg_search", "rg_visit", "rg_gate"],
    )

    xg = c_res.right + GAP_C
    general = L.node(
        "general",
        "general_agent\nqwen2.5:3b · no tools\nsingle call, no ReAct loop\nalways UNGROUNDED",
        "process",
        xg,
        ya,
        230,
    )
    L.node("web", "Tavily API\nand the open web", "cloud", xg, general.bottom + GAP_Y, 210)

    # --- band 4: the fence around the only agent that changes state --------
    ysf = max(c_code.bottom, c_docs.bottom, c_res.bottom, L.nodes["web"].bottom) + GAP_B

    s_path = L.node(
        "s_path",
        "safe_path()\nresolve() BEFORE the check\ncatches ../.. and symlinks",
        "decision",
        MARGIN,
        ysf,
        210,
    )
    s_cmd = L.node(
        "s_cmd",
        "check_command()\nper segment, strictest wins\nunknown = CONFIRM, never ALLOW",
        "decision",
        s_path.right + GAP_X,
        ysf,
        220,
    )
    L.node(
        "s_deny",
        "DENY\nsudo · rm -r · dd · chmod 777\ncurl|sh · fork bomb · .env\nnever becomes a question",
        "inout",
        s_cmd.right + GAP_X,
        ysf,
        240,
    )
    s_int = L.node(
        "s_int",
        "interrupt()\npauses the whole graph\nresumes on the same line",
        "decision",
        MARGIN,
        s_path.bottom + GAP_Y,
        210,
    )
    s_act = L.node(
        "s_act",
        "write the file\nrun it · cwd=PROJECT_ROOT\n60s timeout",
        "process",
        s_int.right + GAP_X,
        s_int.y,
        230,
    )
    L.node(
        "s_declined", "declined\nand the agent is told", "inout", s_act.right + GAP_X, s_int.y, 220
    )
    c_safe = L.cluster(
        "c_safe",
        "tools/safety.py  ·  PROJECT_ROOT = cwd captured at launch",
        ["s_path", "s_cmd", "s_deny", "s_int", "s_act", "s_declined"],
    )

    # --- the spine out: gate, tier, you ------------------------------------
    xsp = c_safe.right + GAP_C
    gate = L.node(
        "gate",
        "evidence gate\ndeterministic · no model call\nany ok Observation\n"
        "whose kind is not 'search'",
        "decision",
        xsp,
        ysf,
        240,
    )
    tiers = L.node(
        "tiers",
        "HIGH  ·  no tag shown\nLOW  ·  thin evidence\n"
        "UNGROUNDED  ·  not verified\ntiers, never percentages",
        "data",
        xsp,
        gate.bottom + GAP_Y,
        250,
    )
    L.node("out", "streamed back to you\ntoken by token", "inout", xsp, tiers.bottom + GAP_Y, 230)

    # --- the reference strip: not steps, so not in the flow ----------------
    yref = max(c_safe.bottom, L.nodes["out"].bottom) + GAP_B

    obs = L.node(
        "obs",
        "Observation (frozen)\nok · detail · content\nsource · metrics{kind,…}\n"
        "render() -> [OK] / [TOOL FAILED]",
        "data",
        MARGIN,
        yref,
        250,
    )
    L.node(
        "state",
        "AssistantState (TypedDict)\nmessages   add_messages\n"
        "observations   operator.add\nconfidence · remaining_steps",
        "data",
        obs.right + GAP_X,
        yref,
        250,
    )
    c_contract = L.cluster(
        "c_contract", "contracts  ·  what the evidence gate reads", ["obs", "state"]
    )

    L.node(
        "ollama",
        "Ollama · localhost:11434\nserves every model here\n"
        "qwen2.5:3b · qwen2.5-coder:7b\nqwen2.5vl:3b · nomic-embed-text",
        "server",
        c_contract.right + GAP_C,
        yref,
        250,
    )
    c_serve = L.cluster("c_serve", "model serving", ["ollama"])

    lf = L.node(
        "lf",
        "Langfuse v2\nCallbackHandler on the graph\nauth_check once, cached\n"
        "every span, plus the confidence score",
        "process",
        c_serve.right + GAP_C,
        yref,
        250,
    )
    L.node("lfdb", "docker compose\nweb + postgres", "store", lf.right + GAP_X, yref, 210)
    c_obs = L.cluster("c_obs", "observability  ·  optional, degrades to a no-op", ["lf", "lfdb"])

    xl = c_obs.right + GAP_C
    lp = L.node("l_proc", "PROCESS\nsomething that runs", "process", xl, yref, 160)
    ld = L.node("l_dec", "DECISION\na branch", "decision", lp.right + GAP_X, yref, 120)
    li = L.node("l_io", "IN / OUT\na command or a refusal", "inout", ld.right + GAP_X, yref, 160)
    lda = L.node("l_data", "DATA\na record, not a step", "data", li.right + GAP_X, yref, 160)
    ls = L.node("l_store", "STORE\non disk", "store", lda.right + GAP_X, yref, 130)
    L.node("l_srv", "SERVER\nlong-running", "server", ls.right + GAP_X, yref, 140)
    L.cluster(
        "c_legend",
        "legend  ·  what each shape means",
        ["l_proc", "l_dec", "l_io", "l_data", "l_store", "l_srv"],
    )

    # Slide the top band into the middle, now that the width is settled.
    width = max(c.right for c in L.clusters.values())
    L.shift(top_nodes, ["c_repl"], (width - c_repl.w) / 2 - c_repl.x)

    # --- edges. Every one carries a label: an unlabelled arrow makes the
    # --- reader guess, and assert_every_edge_labelled() enforces it.
    L.edge("user", "repl", "types a prompt")
    L.edge("repl", "sess", "append the turn", "trace")
    L.edge("repl", "errors", "wrap the turn", "trace")
    L.edge("repl", "route", "one line of input")

    L.edge("route", "d_parse", "a real file path")
    L.edge("route", "c_help", "starts with /")
    L.edge("route", "sup", "a question")

    L.edge("d_parse", "d_deny", "resolves")
    L.edge("d_deny", "d_refused", "a denied name", "stop")
    L.edge("d_deny", "d_ask", "allowed")
    L.edge("d_ask", "d_img", "yes · an image")
    L.edge("d_ask", "d_txt", "yes · a document")

    L.edge("c_ingest", "man", "walk · skip backups, nested repos, credentials", "data")
    L.edge("c_remember", "mem", "stored verbatim", "data")
    L.edge("shutdown", "mem", "session summary", "data")
    L.edge("man", "chunker", "changed files only", "data")
    L.edge("chunker", "embed", "chunks", "data")
    L.edge("mem", "embed", "the summary", "data")
    L.edge("embed", "chroma", "delete-then-add, per source", "data")
    L.edge("chroma", "sup", "recalled once, next session", "back")
    L.edge("c_stats", "lf", "fetch_traces · session or window", "trace")

    L.edge("sup", "cg_read", "code")
    L.edge("sup", "dg", "your documents")
    L.edge("sup", "rg", "the web")
    L.edge("sup", "general", "chat · drafting")
    L.edge("sup", "lf", "every span", "trace")
    L.edge("lf", "lfdb", "spans + scores", "trace")

    L.edge("cg_read", "cg_list", "what is here?")
    L.edge("cg_read", "cg_readf", "read a file")
    L.edge("cg_read", "cg_pw", "wants to write")
    L.edge("cg_read", "cg_pc", "wants to run")
    L.edge("cg_readf", "s_path", "check the path")
    L.edge("cg_pw", "s_path", "check the path")
    L.edge("cg_pc", "s_cmd", "check the command")
    L.edge("s_path", "s_deny", "outside root, or a credential", "stop")
    L.edge("s_path", "s_int", "inside root")
    L.edge("s_cmd", "s_act", "ALLOW · read-only")
    L.edge("s_cmd", "s_int", "CONFIRM")
    L.edge("s_cmd", "s_deny", "DENY", "stop")
    L.edge("s_int", "s_act", "you said yes")
    L.edge("s_int", "s_declined", "you said no", "stop")
    L.edge("s_act", "cg_read", "resumes inside the tool", "back")
    L.edge("cg_read", "cg_write", "evidence, no prose")
    L.edge("cg_write", "cg_gate", "the drafted answer")
    L.edge("cg_gate", "gate", "Observations")

    L.edge("dg", "dg_notes", "tech notes")
    L.edge("dg", "dg_res", "your CV")
    L.edge("dg", "dg_exp", "one query per role")
    L.edge("dg_notes", "embed", "query", "data")
    L.edge("dg_res", "embed", "query", "data")
    L.edge("dg_exp", "embed", "query", "data")
    L.edge("dg", "dg_gate", "best score")
    L.edge("dg_gate", "gate", "Observations")

    L.edge("rg", "rg_search", "find pages")
    L.edge("rg", "rg_visit", "read a page")
    L.edge("rg_search", "web", "snippets only", "data")
    L.edge("rg_visit", "web", "the full page", "data")
    L.edge("rg", "rg_gate", "did a page load?")
    L.edge("rg_gate", "gate", "Observations")
    L.edge("general", "gate", "no Observations at all")

    L.edge("gate", "tiers", "reads ok + kind, never the prose")
    L.edge("tiers", "out", "the tag, or none")

    return L


# ---------------------------------------------------------------------------
# draw.io output
# ---------------------------------------------------------------------------
def to_drawio(L: Layout) -> str:
    """An uncompressed mxfile.

    draw.io normally deflates the model into one base64 blob. Plain XML is
    valid, is what Lucidchart's importer reads, and has the side benefit of
    diffing in review.
    """
    w, h = L.bounds()
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device")
    diagram = ET.SubElement(mxfile, "diagram", name="architecture", id="architecture")
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1200",
        dy="800",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(int(w)),
        pageHeight=str(int(h)),
        math="0",
        shadow="0",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    # Attributes go in the attrib dict, never as keywords: "parent" is also
    # the name of SubElement's own first parameter, so parent="1" as a keyword
    # is a TypeError rather than an XML attribute. mypy caught this; the file
    # it produced beforehand had no parent attributes at all.
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def geo(cell: ET.Element, x: float, y: float, cw: float, ch: float) -> None:
        g = ET.SubElement(
            cell,
            "mxGeometry",
            x=str(round(x)),
            y=str(round(y)),
            width=str(round(cw)),
            height=str(round(ch)),
        )
        g.set("as", "geometry")

    # Clusters first so they sit behind their contents.
    for c in L.clusters.values():
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": c.id,
                "value": c.title,
                "parent": "1",
                "vertex": "1",
                "style": (
                    "rounded=0;whiteSpace=wrap;html=1;fillColor=#f7f9fc;strokeColor=#b7c3d6;"
                    "dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;"
                    f"fontSize={FONT_SIZE};fontColor=#5b6878;"
                ),
            },
        )
        geo(cell, c.x, c.y, c.w, c.h)

    for n in L.nodes.values():
        shape, fill, stroke = KINDS[n.kind]
        # <br>, not \n or &#10;. ElementTree escapes the tag to &lt;br&gt; on
        # the way out, draw.io unescapes it back to <br> on the way in, and
        # html=1 renders it as a line break. Writing &#10; here instead gives
        # &amp;#10; in the file and the entity shows up as literal text.
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": n.id,
                "value": n.label.replace("\n", "<br>"),
                "parent": "1",
                "vertex": "1",
                "style": (
                    f"{shape}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                    f"fontSize={FONT_SIZE};verticalAlign=middle;align=center;"
                ),
            },
        )
        geo(cell, n.x, n.y, n.w, n.h)

    for i, e in enumerate(L.edges):
        # No exitX/entryX. Pinning every edge to leave the bottom and enter
        # the top is right for the spine and wrong for everything else - the
        # sideways hops (a denied file, a tool into safety.py) would loop right
        # round their own box. Floating connections let the router pick the
        # nearest side, which is also what a person expects when they drag a
        # box after importing it.
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"e{i}",
                "value": e.label,
                "parent": "1",
                "edge": "1",
                "source": e.src,
                "target": e.dst,
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;jettySize=auto;"
                    f"orthogonalLoop=1;{EDGES[e.style]}fontSize={FONT_SIZE - 3};"
                    "labelBackgroundColor=#ffffff;"
                ),
            },
        )
        g = ET.SubElement(cell, "mxGeometry", relative="1")
        g.set("as", "geometry")

    ET.indent(mxfile, space="  ")
    return ET.tostring(mxfile, encoding="unicode")


# ---------------------------------------------------------------------------
# preview, so the layout can be checked without opening a diagram tool
# ---------------------------------------------------------------------------
def to_png(L: Layout) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, h = L.bounds()
    img = Image.new("RGB", (int(w), int(h)), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
        efont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE - 3)
    except OSError:  # a preview is not worth a crash
        font = efont = ImageFont.load_default()

    for c in L.clusters.values():
        d.rectangle([c.x, c.y, c.right, c.bottom], outline="#b7c3d6", fill="#f7f9fc", width=2)
        d.text((c.x + 10, c.y + 6), c.title, fill="#5b6878", font=font)

    labels: list[tuple[float, float, str, str]] = []
    for e in L.edges:
        a, b = L.nodes[e.src], L.nodes[e.dst]
        colour = ARROW_COLOURS.get(e.style, "#555555")
        ax, ay = a.x + a.w / 2, a.y + a.h / 2
        bx, by = b.x + b.w / 2, b.y + b.h / 2
        mid = (ay + by) / 2
        d.line([ax, ay, ax, mid, bx, mid, bx, by], fill=colour, width=2)
        # An arrowhead on the last leg, which is vertical into the target.
        tip = by - b.h / 2 if by > mid else by + b.h / 2
        sign = 1 if by > mid else -1
        d.polygon([(bx, tip), (bx - 7, tip - 12 * sign), (bx + 7, tip - 12 * sign)], fill=colour)
        labels.append(((ax + bx) / 2, mid - FONT_SIZE, e.label, colour))

    for n in L.nodes.values():
        _, fill, stroke = KINDS[n.kind]
        d.rectangle([n.x, n.y, n.right, n.bottom], outline=stroke, fill=fill, width=2)
        ty = n.y + PAD_Y + (n.h - 2 * PAD_Y - len(n.lines) * LINE_H) / 2
        for line in n.lines:
            tw = d.textlength(line, font=font)
            d.text((n.x + n.w / 2 - tw / 2, ty), line, fill="#111111", font=font)
            ty += LINE_H

    # Labels last. draw.io paints an edge label over whatever it crosses, so a
    # preview that paints them underneath reports overlaps that will not happen
    # and hides the ones that will.
    for lx, ly, text, colour in labels:
        tw = d.textlength(text, font=efont)
        d.rectangle([lx - tw / 2 - 3, ly - 2, lx + tw / 2 + 3, ly + FONT_SIZE], fill="white")
        d.text((lx - tw / 2, ly), text, fill=colour, font=efont)

    img.save(OUT_PNG)


def main() -> None:
    L = compose()
    L.assert_no_overlap()
    L.assert_clusters_clean()
    L.assert_every_edge_labelled()
    # Trailing newline, so end-of-file-fixer does not rewrite the file on every
    # commit and leave the generator and the hook undoing each other.
    OUT_XML.write_text(to_drawio(L) + "\n")
    to_png(L)
    w, h = L.bounds()
    print(f"wrote {OUT_XML}  ({len(L.nodes)} nodes, {len(L.edges)} edges, {int(w)}x{int(h)})")
    print(f"wrote {OUT_PNG}  (preview)")


if __name__ == "__main__":
    main()
