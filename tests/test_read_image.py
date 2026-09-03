"""read_image.py: downscaling, and catching the model's silent shrug.

An unreadable image does not error - it returns "The image appears to be a
flowchart", fluent and empty. Most of these tests exist to keep that from
reaching the user as an answer.
"""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image
from pydantic import Field

from myassistant import config
from myassistant.tools import read_image

GOOD = (
    "The diagram shows a star schema with a central fact_sales table joined to "
    "dim_store, dim_customer, dim_product, dim_time and dim_sales_type. Each "
    "dimension carries a primary key and several descriptive columns."
)


class _Vision(BaseChatModel):
    reply: str = GOOD
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "vision"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


@pytest.fixture
def png(tmp_path):
    def _make(w=800, h=600, name="d.png"):
        p = tmp_path / name
        Image.new("RGB", (w, h), "white").save(p)
        return p

    return _make


# --- downscaling ---


def test_a_huge_image_is_downscaled_before_it_is_sent(png):
    """Measured: the same diagram was unreadable at 4843x5796 and readable at 1600."""
    obs = read_image.look(png(4843, 5796), model=_Vision())
    assert obs.metrics["original"] == "4843x5796"
    assert max(int(n) for n in obs.metrics["sent"].split("x")) == config.VISION_MAX_EDGE


def test_a_small_image_is_not_upscaled(png):
    obs = read_image.look(png(696, 564), model=_Vision())
    assert obs.metrics["sent"] == "696x564"


def test_aspect_ratio_is_preserved(png):
    obs = read_image.look(png(4000, 1000), model=_Vision())
    w, h = (int(n) for n in obs.metrics["sent"].split("x"))
    assert abs((w / h) - 4.0) < 0.05


def test_a_palette_image_does_not_crash(tmp_path):
    """A PNG saved in palette mode cannot be written back as RGB without convert()."""
    p = tmp_path / "pal.png"
    Image.new("P", (500, 500)).save(p)
    assert read_image.look(p, model=_Vision()).ok


# --- the silent failure ---


def test_a_generic_shrug_is_reported_as_a_failure(png):
    """The exact observed output when a diagram is too large to read."""
    obs = read_image.look(png(), model=_Vision(reply="The image appears to be a flowchart"))
    assert not obs.ok
    assert "could not read this image" in obs.detail


def test_a_claim_of_blindness_is_reported_as_a_failure(png):
    obs = read_image.look(png(), model=_Vision(reply="I'm sorry, but I cannot see the image."))
    assert not obs.ok


def test_a_very_short_answer_is_reported_as_a_failure(png):
    obs = read_image.look(png(), model=_Vision(reply="A chart."))
    assert not obs.ok


def test_the_vague_answer_is_still_returned(png):
    """It is a clue about what went wrong, even though it is not an answer."""
    obs = read_image.look(png(), model=_Vision(reply="The image appears to be a diagram"))
    assert obs.content


def test_a_real_reading_is_ok(png):
    obs = read_image.look(png(), model=_Vision())
    assert obs.ok
    assert "fact_sales" in obs.content
    assert obs.metrics["kind"] == "image"


# --- failure paths ---


def test_a_missing_file_is_an_explicit_failure(tmp_path):
    obs = read_image.look(tmp_path / "nope.png", model=_Vision())
    assert not obs.ok
    assert "no such image" in obs.detail


def test_an_unsupported_format_says_what_it_can_read(tmp_path):
    """HEIC is the realistic case - macOS photos default to it."""
    p = tmp_path / "photo.heic"
    p.write_bytes(b"x")
    obs = read_image.look(p, model=_Vision())
    assert not obs.ok
    assert ".png" in obs.detail


def test_a_corrupt_image_degrades_instead_of_raising(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"not really a png")
    obs = read_image.look(p, model=_Vision())
    assert not obs.ok
    assert "could not open" in obs.detail


def test_a_dead_vision_model_degrades_instead_of_raising(png):
    class _Dead(_Vision):
        def _generate(self, *a, **kw):
            raise ConnectionError("ollama is not running")

    obs = read_image.look(png(), model=_Dead())
    assert not obs.ok
    assert "vision model failed" in obs.detail


def test_a_question_is_passed_through_and_a_default_supplied(png):
    model = _Vision()
    read_image.look(png(), "what services are shown?", model=model)
    assert "what services are shown?" in str(model.seen[0][0].content)

    model2 = _Vision()
    read_image.look(png(), model=model2)
    assert "Read the contents" in str(model2.seen[0][0].content)
