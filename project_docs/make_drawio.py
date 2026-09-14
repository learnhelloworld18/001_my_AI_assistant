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
from itertools import pairwise
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
GAP_Y = 44  # between boxes stacked inside a cluster
GAP_C = 70  # between clusters, horizontally
GAP_B = 84  # between bands, vertically
MARGIN = 40
# A cluster's border sits CPAD + TITLE_H above its first box, so a band that
# starts GAP_B below the last one leaves only GAP_B minus that showing - 13px
# of the intended 84. Bands add it back, so GAP_B is the gap you actually see.
BAND_CHROME = 20 + round(30 * BOX)

# How far a route must stay off a box it is not attached to. Missing a box by
# a pixel still reads as touching it - the run to docs_agent traced the bottom
# edge of /stats and looked connected to it. 16 is the ceiling, not a taste:
# the tightest corridors are GAP_Y=44, and at 20 a side nothing fits through
# them and assert_routes_clear fails outright.
CLEARANCE = 16.0

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
# what an unstyled edge looks like. jumpStyle=arc makes a line hop over the
# ones it crosses instead of merging into them at the junction - with 62 edges
# a plain crossing is ambiguous about which line continues where.
JUMPS = "jumpStyle=arc;jumpSize=10;"
EDGES = {
    "path": "strokeColor=#333333;strokeWidth=2;endArrow=classic;endFill=1;",
    # The routing decision is the one thing the whole graph turns on, so its
    # four hand-offs are drawn heavier than the plumbing around them.
    "route": "strokeColor=#1f3d6e;strokeWidth=5;endArrow=classic;endFill=1;",
    "stop": "strokeColor=#b02c2c;strokeWidth=2;endArrow=classic;endFill=1;",
    "data": "strokeColor=#2f5fbf;strokeWidth=2;endArrow=classic;endFill=1;",
    "back": "strokeColor=#333333;strokeWidth=2;dashed=1;endArrow=classic;endFill=1;",
    "trace": (
        "strokeColor=#777777;strokeWidth=2;dashed=1;dashPattern=1 4;endArrow=open;endFill=0;"
    ),
}
ARROW_COLOURS = {"stop": "#b02c2c", "data": "#2f5fbf", "trace": "#999999", "route": "#1f3d6e"}
STROKE_W = {"route": 5}  # preview line width; everything else is 2


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

    yr = user.bottom + GAP_B + BAND_CHROME
    repl = L.node(
        "repl",
        "prompt_toolkit\nFileHistory ~/.myassistant/history\n"
        "completes only on '/'\nstreams tokens · subgraphs=True",
        "process",
        0,
        yr,
        270,
    )
    # Under prompt_toolkit rather than beside it. Neither of these is a peer of
    # the REPL in the flow - the Session is what it appends to and the
    # try/except is what it runs inside - so a row below reads as "belongs to"
    # where a row beside reads as "comes after".
    sess = L.node(
        "sess",
        "Session\nhistory: list[BaseMessage]\nsession_id groups traces\nrecall runs once per run",
        "data",
        0,
        repl.bottom + GAP_Y,
        250,
    )
    errors = L.node(
        "errors",
        "per-turn try/except\none bad turn never\nkills the loop",
        "inout",
        sess.right + GAP_X,
        sess.y,
        230,
    )
    # Beside the try/except, on the same row: both are ways a run ends rather
    # than steps in a turn, and the top row is then just the thing you type at.
    L.node(
        "shutdown",
        "/exit · Ctrl-D · SIGHUP · SIGTERM\n1. flush Langfuse\n"
        "2. summarise, 20s cap\nsecond signal exits at once",
        "process",
        errors.right + GAP_X,
        sess.y,
        270,
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
    yb = route.bottom + GAP_B + BAND_CHROME

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

    # --- band 3: the agents, one job each ----------------------------------
    ya = max(c_drop.bottom, c_meta.bottom, sup.bottom) + GAP_B + BAND_CHROME

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

    # A band down from the branch row, not level with it. Sitting beside
    # dragged-file / meta-commands / supervisor it read as a fourth answer
    # to "command, file path or question?", which is not what it is: it is
    # what /ingest writes to and what docs_agent reads from, so it belongs
    # level with the agents that use it.
    xs = max(general.right, L.nodes["web"].right) + GAP_C
    man = L.node(
        "man",
        "manifest.db (SQLite)\n(source, collection) -> hash\nhashes CONTENT, not mtime",
        "store",
        xs,
        ya,
        250,
    )
    chunker = L.node(
        "chunker",
        "RecursiveCharacterTextSplitter\n1000 chars · 150 overlap\nrole tagged from path",
        "process",
        man.right + GAP_X,
        ya,
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

    # --- band 4: the fence around the only agent that changes state --------
    ysf = (
        max(c_code.bottom, c_docs.bottom, c_res.bottom, L.nodes["web"].bottom, c_store.bottom)
        + GAP_B
        + BAND_CHROME
    )

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
    yref = max(c_safe.bottom, L.nodes["out"].bottom) + GAP_B + BAND_CHROME

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

    # Slide the top band so the decision sits over the branches it feeds,
    # left edge against meta-commands. Centred on the whole canvas it drifted
    # 910px right of them - the canvas is wide because of the agents band, and
    # the decision has nothing to do with that band.
    L.shift(top_nodes, ["c_repl"], L.clusters["c_meta"].x - route.x)

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

    L.edge("sup", "cg_read", "code", "route")
    L.edge("sup", "dg", "your documents", "route")
    L.edge("sup", "rg", "the web", "route")
    L.edge("sup", "general", "chat · drafting", "route")
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
                    f"orthogonalLoop=1;{JUMPS}{EDGES[e.style]}fontSize={FONT_SIZE - 3};"
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
Point = tuple[float, float]


def _off_titles(mid: float, floor: float, clusters: dict[str, Cluster]) -> float:
    """Lift a crossing run off any cluster title it would otherwise strike out.

    A route's horizontal run sits halfway between the two boxes it joins. When
    the target is the first box inside a cluster, halfway lands in the gap
    between the cluster's top border and that box - which is exactly where the
    cluster's title is written, so the run drew a line straight through the
    text. The fan-out from "what is this line?" did it to all three branch
    titles at once, because they share a band and therefore share the height.

    Halfway is only a default, so the run moves above the cluster instead.
    `floor` is the y it must stay below (the source's own edge); if clearing
    the title would push it past that there is no room, and it stays put.
    """
    for c in clusters.values():
        if c.y - 4 <= mid <= c.y + Layout.TITLE_H + 4:
            lifted = c.y - 10
            if lifted > floor:
                mid = lifted
    return mid


def side_of(a: Node, b: Node) -> tuple[str, str]:
    """Which side an edge leaves a by, and which side it enters b by."""
    if b.y > a.bottom:
        return "S", "N"
    if a.y > b.bottom:
        return "N", "S"
    if b.x > a.right:
        return "E", "W"
    return "W", "E"


def ports(L: Layout) -> dict[tuple[int, str], float]:
    """Spread the edges meeting a box across that side instead of stacking.

    Every edge aimed at the centre of its endpoints, so two edges arriving at
    the same box from the same direction ran down the identical line and drew
    over each other - /remember and the shutdown summary both reach memory.py,
    and the pair looked like one line with a stray branch.

    Each edge gets its own slot across the middle 60% of the side, so they stay
    visibly separate and still clearly belong to the same box.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(L.edges):
        sa, sb = side_of(L.nodes[e.src], L.nodes[e.dst])
        groups.setdefault((e.src, sa), []).append(i)
        groups.setdefault((e.dst, sb), []).append(i)
    out: dict[tuple[int, str], float] = {}
    for (nid, side), idxs in groups.items():
        n = L.nodes[nid]
        span = (n.w if side in "NS" else n.h) * 0.6
        for j, i in enumerate(idxs):
            out[(i, nid)] = ((j + 1) / (len(idxs) + 1) - 0.5) * span
    return out


LANE = 9.0


def _overlap(s: tuple[Point, Point], t: tuple[Point, Point]) -> str | None:
    """'v' or 'h' if these two segments lie along the same line and overlap."""
    (ax0, ay0), (ax1, ay1) = s
    (bx0, by0), (bx1, by1) = t
    if ax0 == ax1 == bx0 == bx1:
        lo1, hi1 = sorted((ay0, ay1))
        lo2, hi2 = sorted((by0, by1))
        return "v" if min(hi1, hi2) - max(lo1, lo2) > 20 else None
    if ay0 == ay1 == by0 == by1:
        lo1, hi1 = sorted((ax0, ax1))
        lo2, hi2 = sorted((bx0, bx1))
        return "h" if min(hi1, hi2) - max(lo1, lo2) > 20 else None
    return None


def separate(routes: list[list[Point]], obstacles: list[Node], skips: list[set[str]]) -> int:
    """Pull apart runs that ended up along the same line.

    Ports fix the stacking at a box's edge, but not this: A* hands two edges
    the same corridor when it is the only clear one, and _tidy clips the port
    offset away at the ends. /remember and the shutdown summary both reach
    memory.py down the identical vertical, so the pair drew as one line with a
    stray branch.

    Only interior segments move. The first and last are attached to a box, and
    sliding them sideways would detach the line from what it connects. A nudge
    is kept only if the route still clears every obstacle, so separating lines
    can never reintroduce a line through a box.
    """
    moves = 0
    for _ in range(4):  # a nudge can create a fresh conflict; settle it
        changed = False
        for i, ri in enumerate(routes):
            for j, rj in enumerate(routes):
                if j <= i:
                    continue
                for si in range(len(ri) - 1):
                    for sj in range(1, len(rj) - 2):  # interior only
                        axis = _overlap((ri[si], ri[si + 1]), (rj[sj], rj[sj + 1]))
                        if axis is None:
                            continue
                        for delta in (LANE, -LANE, 2 * LANE, -2 * LANE):
                            trial = list(rj)
                            for k in (sj, sj + 1):
                                x, y = trial[k]
                                trial[k] = (x + delta, y) if axis == "v" else (x, y + delta)
                            if not _crosses(trial, obstacles, skips[j]):
                                routes[j] = trial
                                rj = trial
                                moves += 1
                                changed = True
                                break
                        break
        if not changed:
            break
    return moves


def obstacles_of(L: Layout) -> list[Node]:
    """Every box, plus every cluster panel as a solid rectangle.

    A cluster is a claim about membership: a line crossing research_agent's
    panel on its way somewhere else reads as one of research_agent's lines.
    The router only knew about boxes, so routes cut straight through panels
    they had nothing to do with. Panels are obstacles now, and _skip() lets an
    edge through the ones its own endpoints live in.
    """
    blockers = [Node(f"#{cid}", "", "process", c.x, c.y, c.w, c.h) for cid, c in L.clusters.items()]
    return [*L.nodes.values(), *blockers]


def _skip(a: Node, b: Node, clusters: dict[str, Cluster]) -> set[str]:
    """The obstacles this edge is allowed to touch: its own ends, and any
    panel one of them sits inside - it has to get out, or in."""
    skip = {a.id, b.id}
    for cid, c in clusters.items():
        for n in (a, b):
            if c.x <= n.x and n.right <= c.right and c.y <= n.y and n.bottom <= c.bottom:
                skip.add(f"#{cid}")
    return skip


def _crosses(route: list[Point], nodes: list[Node], skip: set[str]) -> bool:
    """Does any leg of this route pass through a box that is not its own end?"""
    for (x0, y0), (x1, y1) in pairwise(route):
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        for n in nodes:
            if n.id in skip:
                continue
            if (
                n.x - CLEARANCE < hi_x
                and lo_x < n.right + CLEARANCE
                and n.y - CLEARANCE < hi_y
                and lo_y < n.bottom + CLEARANCE
            ):
                return True
    return False


def _grid_route(
    a: Node,
    b: Node,
    nodes: list[Node],
    skip: set[str],
    pad: float = 16.0,
    pa: float = 0.0,
    pb: float = 0.0,
) -> list[Point] | None:
    """A* over a visibility grid, for the routes a straight slide cannot solve.

    _clear_run only moves the crossing leg up or down. That fails when a third
    box sits squarely between source and target with no vertical room either
    side of it - safe_path() -> DENY has check_command() directly in the way,
    and no height clears it, because the obstacle spans the whole gap.

    Those need to go AROUND, which means more than the three legs the simple
    router emits. The grid is built from the obstacle edges themselves (every
    box's sides, pushed out by `pad`), so it stays small - a few thousand cells
    - and only the handful of edges that actually need it ever get here.

    Turn cost is part of the search rather than a tidy-up afterwards: without
    it A* returns staircases that are the same length as the clean L and look
    like a mistake.
    """
    import heapq

    obstacles = [n for n in nodes if n.id not in skip]
    horizontal = side_of(a, b)[0] in "NS"
    ax = a.x + a.w / 2 + (pa if horizontal else 0.0)
    ay = a.y + a.h / 2 + (0.0 if horizontal else pa)
    bx = b.x + b.w / 2 + (pb if horizontal else 0.0)
    by = b.y + b.h / 2 + (0.0 if horizontal else pb)
    xs, ys = {ax, bx}, {ay, by}
    for n in obstacles:
        xs.update((n.x - pad, n.right + pad))
        ys.update((n.y - pad, n.bottom + pad))
    xv, yv = sorted(xs), sorted(ys)
    xi = {v: i for i, v in enumerate(xv)}
    yi = {v: i for i, v in enumerate(yv)}

    def free(x0: float, y0: float, x1: float, y1: float) -> bool:
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return not any(
            n.x - CLEARANCE < hi_x
            and lo_x < n.right + CLEARANCE
            and n.y - CLEARANCE < hi_y
            and lo_y < n.bottom + CLEARANCE
            for n in obstacles
        )

    start, goal = (xi[ax], yi[ay]), (xi[bx], yi[by])
    # State carries the axis last travelled, so a turn can be charged for.
    begin = (start[0], start[1], -1)
    dist: dict[tuple[int, int, int], float] = {begin: 0.0}
    prev: dict[tuple[int, int, int], tuple[int, int, int] | None] = {begin: None}
    seen: set[tuple[int, int, int]] = set()
    heap: list[tuple[float, tuple[int, int, int]]] = [(0.0, begin)]
    turn_cost = 60.0
    end_state = None
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in seen:
            continue
        seen.add(cur)
        i, j, axis = cur
        if (i, j) == goal:
            end_state = cur
            break
        for di, dj, nax in ((1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < len(xv) and 0 <= nj < len(yv)):
                continue
            if not free(xv[i], yv[j], xv[ni], yv[nj]):
                continue
            step = abs(xv[ni] - xv[i]) + abs(yv[nj] - yv[j])
            cost = dist[cur] + step + (turn_cost if axis != -1 and nax != axis else 0.0)
            nxt = (ni, nj, nax)
            if cost < dist.get(nxt, float("inf")):
                dist[nxt] = cost
                prev[nxt] = cur
                heur = abs(xv[ni] - bx) + abs(yv[nj] - by)
                heapq.heappush(heap, (cost + heur, nxt))
    if end_state is None:
        return None

    path: list[Point] = []
    node: tuple[int, int, int] | None = end_state
    while node is not None:
        path.append((xv[node[0]], yv[node[1]]))
        node = prev[node]
    path.reverse()
    return _tidy(path, a, b)


def _tidy(path: list[Point], a: Node, b: Node) -> list[Point]:
    """Drop collinear points, then clip the ends back to the box borders."""
    merged: list[Point] = [path[0]]
    for p in path[1:]:
        if len(merged) >= 2:
            (x0, y0), (x1, y1) = merged[-2], merged[-1]
            if (x0 == x1 == p[0]) or (y0 == y1 == p[1]):
                merged[-1] = p
                continue
        merged.append(p)

    def clip(pts: list[Point], box: Node) -> list[Point]:
        for i, ((x0, y0), (x1, y1)) in enumerate(pairwise(pts)):
            inside1 = box.x <= x1 <= box.right and box.y <= y1 <= box.bottom
            if inside1:
                continue
            if x0 == x1:
                return [(x0, box.bottom if y1 > y0 else box.y), *pts[i + 1 :]]
            return [(box.right if x1 > x0 else box.x, y0), *pts[i + 1 :]]
        return pts

    merged = clip(merged, a)
    merged = clip(merged[::-1], b)[::-1]
    return merged


def _clear_run(
    mid: float, lo: float, hi: float, x0: float, x1: float, nodes: list[Node], skip: set[str]
) -> float:
    """Slide a crossing run to a height where it does not cut through a box.

    The router picks the two boxes an edge joins and ignores everything in
    between, so a run could cross a third box entirely - /remember -> memory.py
    ran its horizontal leg straight through Chroma. Nodes paint over edges, so
    the line died at Chroma's border and its arrowhead was left stranded on the
    far side, looking like a missing arrow rather than a covered one.

    Searches outward from the halfway point, within the gap the route has to
    work in, and keeps halfway if nothing is clear.
    """
    left, right = min(x0, x1), max(x0, x1)
    boxes = [n for n in nodes if n.id not in skip]

    def crosses(y: float) -> bool:
        return any(
            n.x - CLEARANCE < right
            and left < n.right + CLEARANCE
            and n.y - CLEARANCE < y < n.bottom + CLEARANCE
            for n in boxes
        )

    if not crosses(mid):
        return mid
    for step in range(1, 40):
        for cand in (mid - step * 6, mid + step * 6):
            if lo + 4 <= cand <= hi - 4 and not crosses(cand):
                return cand
    return mid


def _simple_route(
    a: Node,
    b: Node,
    clusters: dict[str, Cluster],
    nodes: list[Node],
    pa: float = 0.0,
    pb: float = 0.0,
) -> list[Point]:
    """An orthogonal route from the EDGE of a to the EDGE of b.

    The previous version ran centre to centre, which drove every line straight
    through the interiors of both boxes. Nodes paint over edges, so the line
    vanished and only a stub survived in the gap - and a label placed at the
    midpoint of that route landed on top of a box's text rather than in the
    space between boxes. Leaving and entering at the edges puts the crossing
    leg in the gap, which is where a label can actually go.
    """
    # The offset runs along the side the edge leaves by: across the box for a
    # vertical departure, down it for a sideways one.
    across = side_of(a, b)[0] in "NS"
    ax = a.x + a.w / 2 + (pa if across else 0.0)
    bx = b.x + b.w / 2 + (pb if across else 0.0)
    ay = a.y + a.h / 2 + (0.0 if across else pa)
    by = b.y + b.h / 2 + (0.0 if across else pb)
    skip = _skip(a, b, clusters)
    if b.y > a.bottom:  # b is below a
        mid = _off_titles((a.bottom + b.y) / 2, a.bottom + 2, clusters)
        mid = _clear_run(mid, a.bottom, b.y, ax, bx, nodes, skip)
        return [(ax, a.bottom), (ax, mid), (bx, mid), (bx, b.y)]
    if a.y > b.bottom:  # b is above a - a feedback edge
        mid = _clear_run((a.y + b.bottom) / 2, b.bottom, a.y, ax, bx, nodes, skip)
        return [(ax, a.y), (ax, mid), (bx, mid), (bx, b.bottom)]
    if b.x > a.right:  # side by side, b to the right
        mid = (a.right + b.x) / 2
        return [(a.right, ay), (mid, ay), (mid, by), (b.x, by)]
    mid = (a.x + b.right) / 2  # side by side, b to the left
    return [(a.x, ay), (mid, ay), (mid, by), (b.right, by)]


def _route(
    a: Node,
    b: Node,
    clusters: dict[str, Cluster],
    nodes: list[Node],
    pa: float = 0.0,
    pb: float = 0.0,
) -> list[Point]:
    """The simple three-leg route, or a searched one when that cuts a box.

    Path-finding only where it is needed: the simple router is right for
    almost every edge here and produces the plain L that reads best, so it
    stays the default and A* is the exception rather than the rule.
    """
    skip = _skip(a, b, clusters)
    route = _simple_route(a, b, clusters, nodes, pa, pb)
    if _crosses(route, nodes, skip):
        found = _grid_route(a, b, nodes, skip, pa=pa, pb=pb)
        if found is not None and len(found) >= 2:
            return found
    return route


HOP_R = 8.0


def _draw_hopped(
    d: object,
    route: list[Point],
    idx: int,
    verticals: list[tuple[int, float, float, float]],
    colour: str,
    width: int = 2,
) -> None:
    """Draw a route, arcing over every other line it crosses.

    Only horizontal legs hop; verticals are drawn straight. If both hopped,
    each crossing would get two bumps and the pair would read worse than a
    plain junction. One side hopping is the convention for the same reason a
    road bridge only needs one deck.
    """
    for (x0, y0), (x1, y1) in pairwise(route):
        if y0 != y1:  # vertical leg - drawn straight, it is the one hopped over
            d.line([x0, y0, x1, y1], fill=colour, width=width)  # type: ignore[attr-defined]
            continue
        lo, hi = min(x0, x1), max(x0, x1)
        cuts = sorted(
            vx
            for j, vx, vy0, vy1 in verticals
            if j != idx and lo + HOP_R < vx < hi - HOP_R and vy0 < y0 < vy1
        )
        if x0 > x1:
            cuts.reverse()
        step = 1.0 if x1 > x0 else -1.0
        at = x0
        for cx in cuts:
            d.line([at, y0, cx - HOP_R * step, y0], fill=colour, width=width)  # type: ignore[attr-defined]
            d.arc(  # type: ignore[attr-defined]
                [cx - HOP_R, y0 - HOP_R, cx + HOP_R, y0 + HOP_R],
                180,
                360,
                fill=colour,
                width=width,
            )
            at = cx + HOP_R * step
        d.line([at, y0, x1, y1], fill=colour, width=width)  # type: ignore[attr-defined]


def _arrowhead(d: object, route: list[Point], colour: str) -> None:
    """A head on the last leg, pointing the way that leg actually travels.

    All four directions, not just "arrives from above" - a third of the edges
    here arrive sideways or from below, and assuming otherwise drew the head
    detached from its own line.
    """
    (x0, y0), (x1, y1) = route[-2], route[-1]
    dx, dy = x1 - x0, y1 - y0
    size, half = 12.0, 7.0
    if abs(dy) >= abs(dx):
        sign = 1.0 if dy > 0 else -1.0
        pts = [(x1, y1), (x1 - half, y1 - size * sign), (x1 + half, y1 - size * sign)]
    else:
        sign = 1.0 if dx > 0 else -1.0
        pts = [(x1, y1), (x1 - size * sign, y1 - half), (x1 - size * sign, y1 + half)]
    d.polygon(pts, fill=colour)  # type: ignore[attr-defined]


def _above_target(
    a: Node, b: Node, clusters: dict[str, Cluster], th: float, route: list[Point]
) -> Point:
    """Where a label belongs if it can go there: at the arrow it describes.

    "starts with /" means something above the meta-commands box and nothing
    at all above the dragged-file box next to it. Left to the generic search
    a branch label lands on the longest leg of its route, which for a fan-out
    is the horizontal run - and that run passes over the SIBLING branches, so
    each label ended up captioning its neighbour.

    When the edge comes from outside the target's cluster, the spot is above
    the cluster rather than above the box, so the label clears the cluster's
    title strip instead of fighting it.

    "Above" is only right when the line ARRIVES from above. general_agent
    reaches the gate from the right, so a spot above the gate was 250px from
    the nearest point of its own line and the label read as floating free of
    the diagram. For a sideways or upward arrival the label sits just back
    along the final approach instead, which is the same idea - next to the
    arrowhead it explains.
    """
    (x0, y0), (x1, y1) = route[-2], route[-1]
    if x0 != x1:  # arrives sideways
        back = 52.0
        return (x1 + back if x0 > x1 else x1 - back), y1 - th - 8
    if y1 < y0:  # arrives from below
        return x1, y1 + 14

    inner: Cluster | None = None
    for c in clusters.values():
        holds = c.x <= b.x and b.right <= c.right and c.y <= b.y and b.bottom <= c.bottom
        if holds and (inner is None or c.w * c.h < inner.w * inner.h):
            inner = c
    if inner is not None:
        from_inside = (
            inner.x <= a.x
            and a.right <= inner.right
            and inner.y <= a.y
            and a.bottom <= inner.bottom
        )
        if not from_inside:
            return b.x + b.w / 2, inner.y - th - 8
    return b.x + b.w / 2, b.y - th - 10


def _label_spot(
    route: list[Point],
    tw: float,
    th: float,
    blocked: list[tuple[float, float, float, float]],
    preferred: Point,
) -> tuple[float, float, bool]:
    """Somewhere on the route where the label lands on nothing.

    Walks every leg, longest first, and at each of several points along it
    tries the line itself and then a set of offsets perpendicular to it -
    sideways off a vertical leg, above and below a horizontal one. Takes the
    first position that clears every box, every cluster title and every label
    already placed.

    Returns the position and whether it actually found a clear one, so main()
    can report the failures rather than let them pass unnoticed: a preview that
    quietly draws a label over a box is the thing this is meant to catch.
    """

    def clear(cx: float, cy: float) -> bool:
        rect = (cx - tw / 2 - 3, cy - 2, cx + tw / 2 + 3, cy + th)
        return not any(
            rect[0] < bx1 and bx0 < rect[2] and rect[1] < by1 and by0 < rect[3]
            for bx0, by0, bx1, by1 in blocked
        )

    # Above whatever the arrow points at, if that is free. Everything below is
    # the fallback for when it is not.
    px, py = preferred
    for dx in (0.0, -tw / 4, tw / 4, -tw / 2, tw / 2):
        if clear(px + dx, py):
            return px + dx, py, True

    legs = sorted(
        pairwise(route),
        key=lambda leg: abs(leg[1][0] - leg[0][0]) + abs(leg[1][1] - leg[0][1]),
        reverse=True,
    )
    for (x0, y0), (x1, y1) in legs:
        vertical = abs(y1 - y0) >= abs(x1 - x0)
        # Perpendicular to the leg: a vertical line has room to its left and
        # right, a horizontal one above and below.
        offsets: list[Point] = [(0.0, 0.0)]
        if vertical:
            for dist in (tw / 2 + 14, tw / 2 + 52):
                offsets += [(dist, 0.0), (-dist, 0.0)]
        else:
            for dist in (th + 10, th + 34):
                offsets += [(0.0, -dist), (0.0, dist)]
        for t in (0.5, 0.35, 0.65, 0.2, 0.8):
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t - th
            for ox, oy in offsets:
                cx, cy = px + ox, py + oy
                rect = (cx - tw / 2 - 3, cy - 2, cx + tw / 2 + 3, cy + th)
                if not any(
                    rect[0] < bx1 and bx0 < rect[2] and rect[1] < by1 and by0 < rect[3]
                    for bx0, by0, bx1, by1 in blocked
                ):
                    return cx, cy, True
    # Nothing on the route itself. That happens for a hop between two boxes
    # standing side by side: the horizontal run is GAP_X wide and the label is
    # several times that, so it spills onto both. Widening GAP_X until every
    # label fits would undo the whole point of tight gaps, so instead walk
    # straight up and down from the midpoint until there is room. The label
    # ends up a little off its line, which reads fine and beats sitting on
    # top of a box's text.
    mid_leg = max(pairwise(route), key=lambda g: abs(g[1][0] - g[0][0]) + abs(g[1][1] - g[0][1]))
    (mx0, my0), (mx1, my1) = mid_leg
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2 - th
    for step in range(1, 20):
        for dy in (-step * (th + 6), step * (th + 6)):
            for dx in (0.0, -tw / 2 - 20, tw / 2 + 20):
                rect = (cx + dx - tw / 2 - 3, cy + dy - 2, cx + dx + tw / 2 + 3, cy + dy + th)
                if not any(
                    rect[0] < bx1 and bx0 < rect[2] and rect[1] < by1 and by0 < rect[3]
                    for bx0, by0, bx1, by1 in blocked
                ):
                    return cx + dx, cy + dy, True
    return cx, cy, False


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

    # Everything a label must not land on: a box, or a cluster's title strip.
    blocked = [n.rect for n in L.nodes.values()]
    blocked += [(c.x, c.y, c.right, c.y + Layout.TITLE_H) for c in L.clusters.values()]

    crowded: list[str] = []
    labels: list[tuple[float, float, float, float, str, str]] = []
    # Routes up front, so each one knows what the others cross.
    port = ports(L)
    obs = obstacles_of(L)
    routes = [
        _route(
            L.nodes[e.src],
            L.nodes[e.dst],
            L.clusters,
            obs,
            port.get((i, e.src), 0.0),
            port.get((i, e.dst), 0.0),
        )
        for i, e in enumerate(L.edges)
    ]
    separate(routes, obs, [_skip(L.nodes[e.src], L.nodes[e.dst], L.clusters) for e in L.edges])
    verticals = [
        (i, x0, min(y0, y1), max(y0, y1))
        for i, r in enumerate(routes)
        for (x0, y0), (x1, y1) in pairwise(r)
        if x0 == x1
    ]
    # Every route as a set of thin rectangles, so a label can be kept off the
    # LINES as well as the boxes. "every span" sat in the channel between two
    # vertical runs and touched both, which reads as if it captions either.
    LINE_T = 7.0
    segs: list[list[tuple[float, float, float, float]]] = [
        [
            (min(x0, x1) - LINE_T, min(y0, y1) - LINE_T, max(x0, x1) + LINE_T, max(y0, y1) + LINE_T)
            for (x0, y0), (x1, y1) in pairwise(r)
        ]
        for r in routes
    ]
    for i, (e, route) in enumerate(zip(L.edges, routes, strict=True)):
        a, b = L.nodes[e.src], L.nodes[e.dst]
        colour = ARROW_COLOURS.get(e.style, "#555555")
        _draw_hopped(d, route, i, verticals, colour, STROKE_W.get(e.style, 2))
        _arrowhead(d, route, colour)
        tw = d.textlength(e.label, font=efont)
        pref = _above_target(a, b, L.clusters, FONT_SIZE, route)
        # Its own line is not an obstacle - a label belongs on it.
        others = [r for j, rects in enumerate(segs) if j != i for r in rects]
        # Panels this edge has no business being inside, for the same reason
        # its line may not cross them: "your documents" had settled in the
        # bottom strip of the meta-commands panel, under /stats, and read as
        # a caption belonging to that panel.
        skip = _skip(a, b, L.clusters)
        panels = [
            (c.x, c.y, c.right, c.bottom) for cid, c in L.clusters.items() if f"#{cid}" not in skip
        ]
        lx, ly, placed = _label_spot(route, tw, FONT_SIZE, blocked + others + panels, pref)
        if not placed:
            crowded.append(f"{e.src}->{e.dst} ({e.label})")
        rect = (lx - tw / 2 - 3, ly - 2, lx + tw / 2 + 3, ly + FONT_SIZE)
        blocked.append(rect)  # so two labels cannot stack on each other either
        labels.append((lx, ly, tw, 0.0, e.label, colour))

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
    for lx, ly, tw, _, text, colour in labels:
        d.rectangle([lx - tw / 2 - 3, ly - 2, lx + tw / 2 + 3, ly + FONT_SIZE], fill="white")
        d.text((lx - tw / 2, ly), text, fill=colour, font=efont)

    img.save(OUT_PNG)
    if crowded:
        print(f"  {len(crowded)} labels had nowhere clear to go: {', '.join(crowded)}")


def assert_routes_clear(L: Layout) -> None:
    """No edge may pass through a box that is not one of its own ends.

    Checked rather than hoped for: a line that disappears behind a box takes
    its arrowhead with it and reads as a missing arrow, which is exactly the
    bug this started as.
    """
    nodes = obstacles_of(L)
    port = ports(L)
    bad = [
        f"{e.src}->{e.dst}"
        for i, e in enumerate(L.edges)
        if _crosses(
            _route(
                L.nodes[e.src],
                L.nodes[e.dst],
                L.clusters,
                nodes,
                port.get((i, e.src), 0.0),
                port.get((i, e.dst), 0.0),
            ),
            nodes,
            _skip(L.nodes[e.src], L.nodes[e.dst], L.clusters),
        )
    ]
    if bad:
        raise AssertionError(f"routes cutting through a box: {bad}")


def main() -> None:
    L = compose()
    L.assert_no_overlap()
    assert_routes_clear(L)
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
