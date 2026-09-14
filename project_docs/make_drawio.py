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

# draw.io renders labels at 12px Helvetica by default. 0.58 * size is a
# deliberately pessimistic average glyph width - overestimating wraps a line
# early, which is harmless, while underestimating pushes text out of its box.
FONT_SIZE = 12
CHAR_W = FONT_SIZE * 0.58
LINE_H = 16
PAD_X, PAD_Y = 12, 10

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

# Edge styles, carrying the same meaning as the graphviz version.
EDGES = {
    "path": "strokeColor=#333333;strokeWidth=2;",
    "stop": "strokeColor=#b02c2c;strokeWidth=2;",
    "data": "strokeColor=#2f5fbf;strokeWidth=2;",
    "back": "strokeColor=#333333;strokeWidth=2;dashed=1;",
    "trace": "strokeColor=#777777;strokeWidth=2;dashed=1;dashPattern=1 4;",
}


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
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Cluster:
    id: str
    title: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class EdgeSpec:
    src: str
    dst: str
    label: str
    style: str


class Layout:
    """Nodes at explicit coordinates, clusters sized from what they contain."""

    TITLE_H = 30
    CPAD = 20

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.clusters: list[Cluster] = []
        self.edges: list[EdgeSpec] = []

    def node(self, nid: str, label: str, kind: str, x: float, y: float, w: float = 250) -> str:
        """Place one box. Height follows from how the caption wraps at width w."""
        lines = _wrap(label, w)
        h = max(52.0, 2 * PAD_Y + len(lines) * LINE_H)
        if kind == "decision":
            # Widen first, then re-wrap: a rhombus wastes its corners, so the
            # caption needs the extra width before its height is decided.
            w *= DECISION_SLACK_W
            lines = _wrap(label, w)
            h = max(70.0, (2 * PAD_Y + len(lines) * LINE_H) * DECISION_SLACK_H)
        self.nodes[nid] = Node(nid, label, kind, x, y, w, h, lines)
        return nid

    def cluster(self, cid: str, title: str, members: list[str]) -> None:
        """Draw a titled backdrop around members, sized to fit them."""
        rects = [self.nodes[m].rect for m in members]
        x0 = min(r[0] for r in rects) - self.CPAD
        y0 = min(r[1] for r in rects) - self.CPAD - self.TITLE_H
        x1 = max(r[2] for r in rects) + self.CPAD
        y1 = max(r[3] for r in rects) + self.CPAD
        self.clusters.append(Cluster(cid, title, x0, y0, x1 - x0, y1 - y0))

    def edge(self, src: str, dst: str, label: str = "", style: str = "path") -> None:
        self.edges.append(EdgeSpec(src, dst, label, style))

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

    def bounds(self) -> tuple[float, float]:
        xs = [c.x + c.w for c in self.clusters] + [n.rect[2] for n in self.nodes.values()]
        ys = [c.y + c.h for c in self.clusters] + [n.rect[3] for n in self.nodes.values()]
        return max(xs) + 40, max(ys) + 40


def compose() -> Layout:
    """The diagram itself. Every label is a real module, model or constant."""
    L = Layout()

    # --- band 0/1: you, and the REPL ------------------------------------
    L.node(
        "user",
        "you\nany directory · the one you\nlaunch from is the project",
        "actor",
        1520,
        40,
        250,
    )

    L.node(
        "repl",
        "prompt_toolkit\nFileHistory ~/.myassistant/history\ncompletes only on '/'\nstreams tokens · subgraphs=True",
        "process",
        1180,
        230,
        280,
    )
    L.node(
        "sess",
        "Session\nhistory: list[BaseMessage]\nsession_id groups traces\nrecall runs once per run",
        "data",
        1500,
        230,
        260,
    )
    L.node(
        "errors", "per-turn try/except\none bad turn never\nkills the loop", "inout", 1800, 230, 240
    )
    L.node(
        "shutdown",
        "/exit · Ctrl-D · SIGHUP · SIGTERM\n1. flush Langfuse\n2. summarise, 20s cap\nsecond signal exits at once",
        "process",
        2080,
        230,
        280,
    )
    L.cluster("c_repl", "REPL  ·  main.py", ["repl", "sess", "errors", "shutdown"])

    # --- band 2: the one branch every line goes through -------------------
    L.node(
        "route", "what is this line?\ncommand / file path / question", "decision", 1480, 470, 260
    )

    # --- band 3, left: a dragged file -------------------------------------
    L.node(
        "d_parse",
        "shlex unescape\n'/a/my\\ file.png'\nresolves to a real file?",
        "decision",
        90,
        740,
        230,
    )
    L.node(
        "d_deny", "denylist\n.env  *.pem  *.key\n~/.ssh  ~/.aws  ~/.gnupg", "decision", 90, 930, 230
    )
    L.node("d_refused", "refused\nno prompt shown", "inout", 430, 950, 220)
    L.node("d_ask", "confirm\nresolved path + size\ndefaults to no", "decision", 90, 1120, 230)
    L.node(
        "d_img",
        "read_image.py\ndownscale to 1600px\nqwen2.5vl:3b · keep_alive 2m\nreply <120 chars = unreadable",
        "process",
        60,
        1320,
        270,
    )
    L.node("d_txt", "ingest loaders\nmd · txt · pdf · docx", "process", 380, 1320, 240)
    L.cluster(
        "c_drop",
        "dragged file  ·  dropped.py",
        ["d_parse", "d_deny", "d_refused", "d_ask", "d_img", "d_txt"],
    )

    # --- band 3, middle: meta-commands ------------------------------------
    L.node("c_help", "/help", "inout", 730, 740, 240)
    L.node("c_clear", "/clear\nwipes history, keeps session_id", "inout", 730, 830, 240)
    L.node("c_ingest", "/ingest <path>\n[notes|resume]", "inout", 730, 950, 240)
    L.node("c_remember", "/remember <text>\nstored verbatim", "inout", 730, 1070, 240)
    L.node("c_stats", "/stats [all|24h|3d]\ndefaults to this session", "inout", 730, 1190, 240)
    L.cluster(
        "c_meta",
        "meta-commands  ·  never reach an agent",
        ["c_help", "c_clear", "c_ingest", "c_remember", "c_stats"],
    )

    # --- band 3, right: the question, and where documents live ------------
    L.node(
        "sup",
        "supervisor\nlanggraph-supervisor\nqwen2.5:3b · keep_alive 30m\nparallel_tool_calls=False\noutput_mode=last_message\nInMemorySaver checkpointer",
        "server",
        1450,
        760,
        300,
    )

    L.node(
        "man",
        "manifest.db (SQLite)\n(source, collection) -> hash\nhashes CONTENT, not mtime",
        "store",
        2000,
        740,
        270,
    )
    L.node(
        "chunker",
        "RecursiveCharacterTextSplitter\n1000 chars · 150 overlap\nrole tagged from path",
        "process",
        2320,
        740,
        270,
    )
    L.node(
        "mem",
        "rag/memory.py\nsummaries, never transcripts\nrecall threshold 0.18\n(documents use 0.37)",
        "process",
        2000,
        930,
        270,
    )
    L.node("embed", "nomic-embed-text\nOllamaEmbeddings", "process", 2320, 930, 270)
    L.node(
        "chroma",
        "Chroma\ntech_notes\nresume_interview\nconversation_memory",
        "store",
        2160,
        1120,
        270,
    )
    L.cluster(
        "c_store",
        "storage  ·  ~/.myassistant  ·  embedded, no server",
        ["man", "chunker", "mem", "embed", "chroma"],
    )

    # --- band 4: the agents, one job each ---------------------------------
    L.node(
        "cg_read",
        "read node\nqwen2.5:3b + tools\nreturns evidence, no prose",
        "process",
        90,
        1640,
        250,
    )
    L.node("cg_list", "list_project_files", "process", 90, 1810, 230)
    L.node("cg_readf", "read_project_file\n20k char cap", "process", 350, 1810, 230)
    L.node("cg_pw", "propose_write", "process", 90, 1920, 230)
    L.node("cg_pc", "propose_command", "process", 350, 1920, 230)
    L.node(
        "cg_write", "write node\nqwen2.5-coder:7b-q4_K_M\nno tools bound", "process", 90, 2040, 250
    )
    L.node("cg_gate", "gate", "decision", 390, 2040, 150)
    L.cluster("c_tools", "tools/coding.py", ["cg_list", "cg_readf", "cg_pw", "cg_pc"])
    L.cluster(
        "c_code",
        "coding_agent  ·  read then write  ·  the coder model cannot call tools",
        ["cg_read", "cg_list", "cg_readf", "cg_pw", "cg_pc", "cg_write", "cg_gate"],
    )

    L.node("dg", "docs_agent\nprompt: never fill a gap\nfrom memory", "process", 700, 1640, 250)
    L.node("dg_notes", "search_notes", "process", 700, 1810, 220)
    L.node("dg_res", "search_resume", "process", 950, 1810, 220)
    L.node(
        "dg_exp",
        "search_experience\none query per CAREER_ROLE\ncoverage, not ranking",
        "process",
        700,
        1920,
        250,
    )
    L.node("dg_gate", "gate\ntop_score >= 0.37", "decision", 990, 1930, 170)
    L.cluster(
        "c_docs",
        "docs_agent  ·  qwen2.5:3b  ·  your own documents",
        ["dg", "dg_notes", "dg_res", "dg_exp", "dg_gate"],
    )

    L.node("rg", "research_agent\nprompt: snippets are\nnot evidence", "process", 1330, 1640, 250)
    L.node("rg_search", "web_search · Tavily\nmax 5 · kind=search", "process", 1330, 1810, 230)
    L.node(
        "rg_visit",
        "visit_webpage\nBeautifulSoup + markdownify\nlooks_empty(): <400 chars,\nconsent walls, JS shells",
        "process",
        1590,
        1810,
        250,
    )
    L.node("rg_gate", "gate\nneeds an ok kind=page", "decision", 1330, 1970, 180)
    L.cluster(
        "c_res",
        "research_agent  ·  qwen2.5:3b  ·  the open web",
        ["rg", "rg_search", "rg_visit", "rg_gate"],
    )

    L.node(
        "general",
        "general_agent\nqwen2.5:3b · no tools\nsingle call, no ReAct loop\nalways UNGROUNDED",
        "process",
        1950,
        1640,
        250,
    )
    L.node("web", "Tavily API\nand the open web", "cloud", 1950, 1860, 230)

    # --- band 5: the fence around the only agent that changes state -------
    L.node(
        "s_path",
        "safe_path()\nresolve() BEFORE the check\ncatches ../.. and symlinks",
        "decision",
        90,
        2250,
        230,
    )
    L.node(
        "s_cmd",
        "check_command()\nper segment, strictest wins\nunknown = CONFIRM, never ALLOW",
        "decision",
        430,
        2250,
        240,
    )
    L.node(
        "s_deny",
        "DENY\nsudo · rm -r · dd · chmod 777\ncurl|sh · fork bomb · .env\nnever becomes a question",
        "inout",
        800,
        2270,
        260,
    )
    L.node(
        "s_int",
        "interrupt()\npauses the whole graph\nresumes on the same line",
        "decision",
        90,
        2510,
        230,
    )
    L.node(
        "s_act", "write the file\nrun it · cwd=PROJECT_ROOT\n60s timeout", "process", 430, 2530, 250
    )
    L.node("s_declined", "declined\nand the agent is told", "inout", 800, 2530, 240)
    L.cluster(
        "c_safe",
        "tools/safety.py  ·  PROJECT_ROOT = cwd captured at launch",
        ["s_path", "s_cmd", "s_deny", "s_int", "s_act", "s_declined"],
    )

    # --- band 6-8: the gate, the tier, and back to you --------------------
    L.node(
        "gate",
        "evidence gate\ndeterministic · no model call\nany ok Observation\nwhose kind is not 'search'",
        "decision",
        1450,
        2300,
        260,
    )
    L.node(
        "tiers",
        "HIGH  ·  no tag shown\nLOW  ·  thin evidence\nUNGROUNDED  ·  not verified\ntiers, never percentages",
        "data",
        1450,
        2620,
        280,
    )
    L.node("out", "streamed back to you\ntoken by token", "inout", 1470, 2830, 250)

    # --- the reference strip: not steps, so not in the flow ---------------
    L.node(
        "obs",
        "Observation (frozen)\nok · detail · content\nsource · metrics{kind,…}\nrender() -> [OK] / [TOOL FAILED]",
        "data",
        90,
        3080,
        270,
    )
    L.node(
        "state",
        "AssistantState (TypedDict)\nmessages   add_messages\nobservations   operator.add\nconfidence · remaining_steps",
        "data",
        400,
        3080,
        270,
    )
    L.cluster("c_contract", "contracts  ·  what the evidence gate reads", ["obs", "state"])

    L.node(
        "ollama",
        "Ollama · localhost:11434\nserves every model here\nqwen2.5:3b · qwen2.5-coder:7b\nqwen2.5vl:3b · nomic-embed-text",
        "server",
        790,
        3080,
        280,
    )
    L.cluster("c_serve", "model serving", ["ollama"])

    L.node(
        "lf",
        "Langfuse v2\nCallbackHandler on the graph\nauth_check once, cached\nevery span, plus the confidence score",
        "process",
        1140,
        3080,
        270,
    )
    L.node("lfdb", "docker compose\nweb + postgres", "store", 1450, 3080, 230)
    L.cluster("c_obs", "observability  ·  optional, degrades to a no-op", ["lf", "lfdb"])

    L.node("l_proc", "PROCESS\nsomething that runs", "process", 1790, 3080, 180)
    L.node("l_dec", "DECISION\na branch", "decision", 2010, 3070, 140)
    L.node("l_io", "IN / OUT\na command or a refusal", "inout", 2260, 3080, 180)
    L.node("l_data", "DATA\na record, not a step", "data", 2480, 3080, 180)
    L.node("l_store", "STORE\non disk", "store", 2700, 3080, 150)
    L.node("l_srv", "SERVER\nlong-running", "server", 2900, 3080, 160)
    L.cluster(
        "c_legend",
        "legend  ·  what each shape means",
        ["l_proc", "l_dec", "l_io", "l_data", "l_store", "l_srv"],
    )

    # --- edges ------------------------------------------------------------
    L.edge("user", "repl", "types a prompt")
    L.edge("repl", "route")
    L.edge("repl", "sess", "", "trace")
    L.edge("repl", "errors", "", "trace")

    L.edge("route", "d_parse", "a real file path")
    L.edge("route", "c_help", "starts with /")
    L.edge("route", "sup", "a question")

    L.edge("d_parse", "d_deny")
    L.edge("d_deny", "d_refused", "denied name", "stop")
    L.edge("d_deny", "d_ask")
    L.edge("d_ask", "d_img", "image")
    L.edge("d_ask", "d_txt", "text")

    L.edge("c_ingest", "man", "walk · skip backups,\nnested repos, credentials", "data")
    L.edge("c_remember", "mem", "", "data")
    L.edge("shutdown", "mem", "session summary", "data")
    L.edge("man", "chunker", "changed files only", "data")
    L.edge("chunker", "embed", "", "data")
    L.edge("mem", "embed", "", "data")
    L.edge("embed", "chroma", "delete-then-add per source", "data")
    L.edge("chroma", "sup", "recalled once, next session", "back")
    L.edge("c_stats", "lf", "fetch_traces", "trace")

    L.edge("sup", "cg_read")
    L.edge("sup", "dg")
    L.edge("sup", "rg")
    L.edge("sup", "general")
    L.edge("sup", "lf", "", "trace")
    L.edge("lf", "lfdb", "", "trace")

    L.edge("cg_read", "cg_list")
    L.edge("cg_read", "cg_readf")
    L.edge("cg_read", "cg_pw")
    L.edge("cg_read", "cg_pc")
    L.edge("cg_readf", "s_path")
    L.edge("cg_pw", "s_path")
    L.edge("cg_pc", "s_cmd")
    L.edge("s_path", "s_deny", "outside root,\nor a credential", "stop")
    L.edge("s_path", "s_int")
    L.edge("s_cmd", "s_act", "ALLOW  read-only")
    L.edge("s_cmd", "s_int", "CONFIRM")
    L.edge("s_cmd", "s_deny", "DENY", "stop")
    L.edge("s_int", "s_act", "yes")
    L.edge("s_int", "s_declined", "no", "stop")
    L.edge("s_act", "cg_read", "resumes inside the tool", "back")
    L.edge("cg_read", "cg_write")
    L.edge("cg_write", "cg_gate")
    L.edge("cg_gate", "gate")

    L.edge("dg", "dg_notes")
    L.edge("dg", "dg_res")
    L.edge("dg", "dg_exp")
    L.edge("dg_notes", "embed", "", "data")
    L.edge("dg_res", "embed", "", "data")
    L.edge("dg_exp", "embed", "", "data")
    L.edge("dg", "dg_gate")
    L.edge("dg_gate", "gate")

    L.edge("rg", "rg_search")
    L.edge("rg", "rg_visit")
    L.edge("rg_search", "web", "", "data")
    L.edge("rg_visit", "web", "", "data")
    L.edge("rg", "rg_gate")
    L.edge("rg_gate", "gate")
    L.edge("general", "gate")

    L.edge("gate", "tiers")
    L.edge("tiers", "out")

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
    for c in L.clusters:
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
        style = (
            f"{shape}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontSize={FONT_SIZE};verticalAlign=middle;align=center;"
        )
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
                "style": style,
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
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;jettySize=auto;"
            "orthogonalLoop=1;"
            f"{EDGES[e.style]}fontSize={FONT_SIZE - 1};"
        )
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"e{i}",
                "value": e.label.replace("\n", " "),
                "parent": "1",
                "edge": "1",
                "source": e.src,
                "target": e.dst,
                "style": style,
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
    scale = 2  # retina, so the caption text is legible when read back
    img = Image.new("RGB", (int(w * scale), int(h * scale)), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE * scale)
        tfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", (FONT_SIZE + 1) * scale)
    except OSError:  # a preview is not worth a crash
        font = tfont = ImageFont.load_default()

    for c in L.clusters:
        d.rectangle(
            [c.x * scale, c.y * scale, (c.x + c.w) * scale, (c.y + c.h) * scale],
            outline="#b7c3d6",
            fill="#f7f9fc",
            width=scale,
        )
        d.text(
            (c.x * scale + 10 * scale, c.y * scale + 6 * scale), c.title, fill="#5b6878", font=tfont
        )

    for e in L.edges:
        a, b = L.nodes[e.src], L.nodes[e.dst]
        colour = {"stop": "#b02c2c", "data": "#2f5fbf", "trace": "#999999"}.get(e.style, "#555555")
        ax, ay = (a.x + a.w / 2) * scale, (a.y + a.h / 2) * scale
        bx, by = (b.x + b.w / 2) * scale, (b.y + b.h / 2) * scale
        d.line([ax, ay, ax, (ay + by) / 2, bx, (ay + by) / 2, bx, by], fill=colour, width=scale)

    for n in L.nodes.values():
        _, fill, stroke = KINDS[n.kind]
        d.rectangle(
            [n.x * scale, n.y * scale, (n.x + n.w) * scale, (n.y + n.h) * scale],
            outline=stroke,
            fill=fill,
            width=scale,
        )
        ty = (n.y + PAD_Y) * scale + (n.h - 2 * PAD_Y - len(n.lines) * LINE_H) * scale / 2
        for line in n.lines:
            tw = d.textlength(line, font=font)
            d.text(((n.x + n.w / 2) * scale - tw / 2, ty), line, fill="#111111", font=font)
            ty += LINE_H * scale

    img.save(OUT_PNG)


def main() -> None:
    L = compose()
    L.assert_no_overlap()
    # Trailing newline, so end-of-file-fixer does not rewrite the file on every
    # commit and leave the generator and the hook undoing each other.
    OUT_XML.write_text(to_drawio(L) + "\n")
    to_png(L)
    w, h = L.bounds()
    print(f"wrote {OUT_XML}  ({len(L.nodes)} nodes, {len(L.edges)} edges, {int(w)}x{int(h)})")
    print(f"wrote {OUT_PNG}  (preview)")


if __name__ == "__main__":
    main()
