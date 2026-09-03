"""Read an image - a screenshot, a diagram, a photo of a whiteboard.

    read_image(path, question)
       |
       v
    downscale to VISION_MAX_EDGE      <- not optional, see below
       |
       v
    vision model answers the question
       |
       v
    too short/generic?  ->  ok=False  <- the silent failure, made loud
       |
       v
    Observation

Two findings from testing three models against a diagram whose contents were
known, both of which shape this module:

**Size, not detail, is what breaks it.** The same architecture diagram returned
35 useless characters at 4843x5796 *and* at 2898x2421, then 2049 characters of
real content at 1600px. Vision models downsample to a fixed budget, so a large
image arrives as an illegible smudge. Downscaling first is the whole fix; tiling
was tried and is not needed.

**It fails silently.** An unreadable image does not error. It returns "The image
appears to be a flowchart" - fluent, confident, and empty. Nothing in that says
"I could not read this", which is exactly the failure this project keeps
running into, so it gets the same treatment as a redirect-shell web page: a
crude length check that turns the quiet failure into a loud one.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from myassistant import config
from myassistant.tools.observation import Observation, failed

# What Pillow can open and a vision model can use. HEIC is absent: macOS photos
# are HEIC by default but decoding needs a separate plugin, so it is a
# deliberate gap rather than an oversight.
READABLE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# Phrases a vision model produces when it cannot actually read an image. Length
# alone catches most of it; these catch a wordier shrug.
_SHRUGS = (
    "appears to be a flowchart",
    "appears to be a diagram",
    "cannot provide a description",
    "not visible to me",
    "unable to view",
    "i cannot see",
)


def _downscale(path: Path, max_edge: int) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Shrink to fit max_edge. Returns (png bytes, original size, sent size)."""
    from PIL import Image

    with Image.open(path) as image:
        original = image.size
        # Convert first: a palette or CMYK image cannot be saved as RGB PNG.
        rgb = image.convert("RGB")
        rgb.thumbnail((max_edge, max_edge))  # preserves aspect ratio, never upscales
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue(), original, rgb.size


def _model() -> ChatOllama:
    """The vision model, unloaded quickly - see config.VISION_KEEP_ALIVE."""
    return ChatOllama(
        model=config.VISION_MODEL,
        base_url=config.OLLAMA_HOST,
        keep_alive=config.VISION_KEEP_ALIVE,
    )


def look(path: Path, question: str = "", *, model: Any | None = None) -> Observation:
    """Answer a question about one image. Pure: no graph, no tool plumbing."""
    if not path.exists():
        return failed(f"no such image: {path}", source=str(path), kind="image")
    if path.suffix.lower() not in READABLE:
        return failed(
            f"cannot read {path.suffix} images (readable: {', '.join(sorted(READABLE))})",
            source=str(path),
            kind="image",
        )

    try:
        png, original, sent = _downscale(path, config.VISION_MAX_EDGE)
    except Exception as e:  # noqa: BLE001 - a corrupt file must not kill the turn
        return failed(f"could not open the image: {e}", source=str(path), kind="image")

    prompt = question.strip() or "Read the contents of this image and describe what it shows."
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64.b64encode(png).decode()}",
            },
        ]
    )

    try:
        answer = str((model or _model()).invoke([message]).content).strip()
    except Exception as e:  # noqa: BLE001 - Ollama down, model missing
        return failed(f"vision model failed: {e}", source=str(path), kind="image")

    metrics = {
        "kind": "image",
        "original": f"{original[0]}x{original[1]}",
        "sent": f"{sent[0]}x{sent[1]}",
        "chars": len(answer),
    }

    # The silent failure, caught. A model that cannot read an image still
    # answers - briefly, and about the shape of the thing rather than its
    # contents. Reporting that as a failure is the difference between the model
    # knowing it learned nothing and the model believing it did.
    lowered = answer.lower()
    if len(answer) < config.VISION_MIN_USEFUL_CHARS or any(s in lowered for s in _SHRUGS):
        return Observation(
            ok=False,
            detail=(
                "the model could not read this image - it may be too detailed even "
                "downscaled, or too low-contrast"
            ),
            content=answer,  # kept: a vague answer is still a clue
            source=str(path),
            metrics=metrics,
        )

    return Observation(
        ok=True,
        detail=f"read {path.name}",
        content=answer,
        source=str(path),
        metrics=metrics,
    )
