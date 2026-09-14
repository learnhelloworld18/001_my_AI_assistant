"""Version 2 of the architecture diagram: the v1 layout, with icons.

    uv run python project_docs/make_drawio_v2.py
      -> architecture_v2.drawio
      -> architecture_v2_layout.png

v1 (`make_drawio.py`) is frozen - see tests/test_diagram_locked.py, which
fails if its bytes move. This file does not edit it. It imports v1's engine
whole (layout, routing, A*, label placement, line separation) and changes only
what v2 changes, so an improvement to routing made here is a deliberate
decision to unfreeze v1, not an accident.

How the content is reused
-------------------------
`v1.compose()` builds the whole diagram through `Layout`, which it looks up on
its own module. Swapping that attribute for `IconLayout` before calling it
means v2 gets every box, edge and label of v1 without copying four hundred
lines of content - and stays in step with v1 by construction rather than by
anyone remembering.

Why not mxgraph stencils
------------------------
The obvious way to add icons is draw.io's stencil library
(`shape=mxgraph.azure.python`). It was rejected on the same ground the whole
generator exists for: a stencil shape IS the icon, so its label renders
OUTSIDE the box. That is exactly the `labelloc=b` behaviour that made every
overlap in the graphviz version, and it would also throw away the shape
vocabulary - a decision has to stay a rhombus.

Inline SVG as a data URI instead. Each icon is a few hundred bytes rather than
the ~20KB a PNG from the `diagrams` package costs; mxGraph repeats the URI per
cell, so PNGs would have added roughly a megabyte to a committed file that
pre-commit caps at 500KB. Measured, not assumed.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT_XML = ROOT / "architecture_v2.drawio"
OUT_PNG = ROOT / "architecture_v2_layout.png"


def _load_v1():
    """Import the frozen generator by path - project_docs/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "make_drawio_v1", Path(__file__).parent / "make_drawio.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_drawio_v1"] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1()

# v1's own types, as Any. The generator is loaded by path because project_docs
# is not a package, and pre-commit runs mypy in an isolated venv with no route
# to it either way - so `v1.Layout` is a name mypy cannot resolve, however it
# is spelled. Annotating the real class would fail the hook; annotating Any is
# at least honest about what is checkable here.
LayoutT = Any
NodeT = Any

# Room reserved at the left of a box for its icon. The caption wraps in what is
# left, so an icon never pushes text out of its own box.
GUTTER = 46.0
ICON = 26.0


def _svg(body: str, colour: str) -> str:
    """One icon as a data URI. 24x24, single colour, no fill unless asked."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{colour}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# Line art rather than glyphs: it stays legible at 26px, reads on a pale fill,
# and survives being printed. Drawn from primitives so the whole set is a few
# hundred bytes, not a few hundred kilobytes.
SHAPES = {
    # a module that runs - a terminal window
    "process": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 9l3 3-3 3M13 15h5"/>',
    # a branch - one line splitting into two
    "decision": '<path d="M6 3v6a3 3 0 003 3h6M6 21v-6"/><circle cx="6" cy="3" r="1.6"/>'
    '<circle cx="18" cy="12" r="1.6"/><circle cx="6" cy="21" r="1.6"/>',
    # something you type, or are told
    "inout": '<path d="M4 17l5-5-5-5M12 19h8"/>',
    # a record
    "data": '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>'
    '<path d="M14 2v6h6M9 13h6M9 17h6"/>',
    # persisted on disk
    "store": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
    '<path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    # a long-running process
    "server": '<rect x="2" y="3" width="20" height="7" rx="1"/><rect x="2" y="14" width="20" height="7" rx="1"/>'
    '<path d="M6 6.5h.01M6 17.5h.01"/>',
    # you
    "actor": '<path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    # the open web
    "cloud": '<path d="M18 10h-1.3A7 7 0 104 15.9"/><path d="M8 17a4 4 0 010-8 5 5 0 019.6 1.3A4 4 0 0117 17z"/>',
    # a model
    "model": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9z"/>'
    '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>',
    # a refusal
    "stop": '<circle cx="12" cy="12" r="9"/><path d="M6.5 6.5l11 11"/>',
    # a safety check
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    # search
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4.3-4.3"/>',
}

# Which icon each box gets. Named per box rather than per kind: the shape
# already says what kind a box is, so the icon is free to say what it *does* -
# a magnifying glass on the search tools, a shield on the safety checks.
ICON_FOR: dict[str, str] = {
    "user": "actor",
    "repl": "process",
    "sess": "data",
    "errors": "inout",
    "shutdown": "process",
    "route": "decision",
    "d_parse": "decision",
    "d_deny": "shield",
    "d_ask": "decision",
    "d_img": "model",
    "d_txt": "process",
    "d_refused": "stop",
    "c_help": "inout",
    "c_clear": "inout",
    "c_ingest": "inout",
    "c_remember": "inout",
    "c_stats": "inout",
    "sup": "server",
    "cg_read": "model",
    "cg_write": "model",
    "cg_gate": "decision",
    "cg_list": "search",
    "cg_readf": "search",
    "cg_pw": "process",
    "cg_pc": "process",
    "dg": "model",
    "dg_notes": "search",
    "dg_res": "search",
    "dg_exp": "search",
    "dg_gate": "decision",
    "rg": "model",
    "rg_search": "search",
    "rg_visit": "search",
    "rg_gate": "decision",
    "general": "model",
    "web": "cloud",
    "s_path": "shield",
    "s_cmd": "shield",
    "s_int": "decision",
    "s_act": "process",
    "s_deny": "stop",
    "s_declined": "stop",
    "man": "store",
    "chunker": "process",
    "mem": "process",
    "embed": "model",
    "chroma": "store",
    "gate": "decision",
    "tiers": "data",
    "out": "inout",
    "obs": "data",
    "state": "data",
    "ollama": "server",
    "lf": "process",
    "lfdb": "store",
    "stack_lc": "process",
    "stack_lg": "process",
    "stack_py": "data",
}


# Same reason as LayoutT above: the base class is resolved at runtime, so mypy
# sees a subclass of Any. Nothing here depends on it being checked.
_Base: Any = v1.Layout


class IconLayout(_Base):
    """v1's Layout, with room reserved for an icon on the boxes that get one."""

    def node(
        self,
        nid: str,
        label: str,
        kind: str,
        x: float,
        y: float,
        w: float = 250,
        align: str = "center",
    ) -> NodeT:
        if nid in ICON_FOR:
            # Left-aligned, because an icon on the left and centred text give
            # the box two competing left edges. With the icon there, ragged
            # centring is what looks wrong, not the alignment.
            align = "left"
        n = super().node(nid, label, kind, x, y, w, align)
        if nid in ICON_FOR:
            # Widen AFTER wrapping, so the caption keeps the width it was
            # written for and the gutter is genuinely empty. Widening first
            # would let the text spread under the icon.
            n.w += GUTTER
        return n


def compose() -> LayoutT:
    """v1's diagram, laid out with icon gutters.

    Swapping the Layout class is what makes this the same diagram rather than
    a copy of it: compose() looks Layout up on its own module, so every box it
    creates comes back with the gutter already accounted for, and every
    position computed from those boxes accounts for it too.
    """
    original = v1.Layout
    try:
        v1.Layout = IconLayout  # type: ignore[misc]
        return v1.compose()
    finally:
        v1.Layout = original  # type: ignore[misc]


def to_drawio(L: LayoutT) -> str:
    """v1's writer, plus one image cell per icon.

    The icon is a separate cell rather than part of the box's style, because
    draw.io's `shape=label` (the one style that puts an image beside text)
    would replace the shape - and a decision has to stay a rhombus. A second
    cell keeps both.
    """
    xml = v1.to_drawio(L)
    root = ET.fromstring(xml)
    cells = root.find(".//root")
    assert cells is not None

    by_id = {c.get("id"): c for c in cells.findall("mxCell")}
    for n in L.nodes.values():
        name = ICON_FOR.get(n.id)
        if name is None:
            continue
        _, _, stroke = v1.KINDS[n.kind]
        # v1 writes spacingLeft=14 for a left-aligned box. Push the text past
        # the gutter instead, so the icon sits in space nothing else uses.
        box = by_id.get(n.id)
        if box is not None:
            box.set(
                "style",
                box.get("style", "").replace("spacingLeft=14;", f"spacingLeft={round(GUTTER)};"),
            )
        cell = ET.SubElement(
            cells,
            "mxCell",
            {
                "id": f"icon_{n.id}",
                "value": "",
                "parent": "1",
                "vertex": "1",
                "style": (
                    f"shape=image;image={_svg(SHAPES[name], stroke)};"
                    "imageAspect=1;noLabel=1;editableCssRules=.*;"
                    "strokeColor=none;fillColor=none;"
                ),
            },
        )
        geo = ET.SubElement(
            cell,
            "mxGeometry",
            x=str(round(n.x + (GUTTER - ICON) / 2 + 6)),
            y=str(round(n.y + n.h / 2 - ICON / 2)),
            width=str(round(ICON)),
            height=str(round(ICON)),
        )
        geo.set("as", "geometry")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def to_png(L: LayoutT) -> None:
    """The preview, with a marker where each icon will be.

    A placeholder, not a rendering: Pillow does not draw SVG, and the preview
    exists to check the LAYOUT - that the gutter is reserved and nothing has
    moved into it. What the icon looks like is draw.io's answer, not this
    file's.
    """
    # PAD_X is what v1's preview insets left-aligned text by, and every
    # left-aligned box in v2 is an icon box, so borrowing it for the gutter
    # makes the preview agree with what draw.io will render. Heights were
    # already decided at compose time, so nothing reflows.
    original_out, original_pad = v1.OUT_PNG, v1.PAD_X
    try:
        v1.OUT_PNG = OUT_PNG  # type: ignore[misc]
        v1.PAD_X = GUTTER  # type: ignore[misc]
        v1.to_png(L)
    finally:
        v1.OUT_PNG = original_out  # type: ignore[misc]
        v1.PAD_X = original_pad  # type: ignore[misc]

    from PIL import Image, ImageDraw

    img = Image.open(OUT_PNG).convert("RGB")
    d = ImageDraw.Draw(img)
    for n in L.nodes.values():
        if n.id not in ICON_FOR:
            continue
        _, _, stroke = v1.KINDS[n.kind]
        cx = n.x + (GUTTER - ICON) / 2 + 6 + ICON / 2
        cy = n.y + n.h / 2
        d.ellipse(
            [cx - ICON / 2, cy - ICON / 2, cx + ICON / 2, cy + ICON / 2], outline=stroke, width=3
        )
    img.save(OUT_PNG)


def main() -> None:
    L = compose()
    L.assert_no_overlap()
    L.assert_clusters_clean()
    L.assert_every_edge_labelled()
    v1.assert_routes_clear(L)
    OUT_XML.write_text(to_drawio(L) + "\n")
    to_png(L)
    w, h = L.bounds()
    icons = sum(1 for n in L.nodes if n in ICON_FOR)
    print(f"wrote {OUT_XML}  ({len(L.nodes)} nodes, {icons} icons, {int(w)}x{int(h)})")
    print(f"wrote {OUT_PNG}  (preview)")


if __name__ == "__main__":
    main()
