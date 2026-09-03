"""The two RAG tools - one per collection, deliberately.

One tool that takes a collection argument would be fewer lines. But then a 3B
model has to fill in that argument correctly, and choosing between two named
tools is a decision small models make far more reliably than choosing a string
value. The tool name *is* the routing decision.

The descriptions say what each collection holds in the words a question would
use, and say when not to use it - that is the lever on whether the tool gets
picked at all, and the wikipedia_search-never-chosen lesson from the prior
build is what it exists to avoid.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from myassistant.rag.query import search, search_across_roles
from myassistant.rag.store import Collection
from myassistant.tools.observation import emit


@tool
def search_notes(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search the user's own technical notes and reference material.

    Their saved notes on tools and concepts - Spark, Kafka, cloud services,
    architecture patterns - written or collected by them. Use for "what do my
    notes say about X" and for technical questions where their own material is
    likely to be more specific than a web page.

    Not for their CV, job applications or interview answers: that is
    search_resume.

    Args:
        query: what to look for, in plain words
    """
    return emit(search(Collection.TECH_NOTES, query), tool_call_id)


@tool
def search_experience(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Summarise the user's experience across ALL of their data engineering roles.

    Use for any question spanning more than one job: "walk me through my
    career", "what's my experience with X across roles", "summarise my
    background", "tell me about yourself". Searches each role separately so no
    employer is left out, and returns them most recent first.

    For a question about one specific company, use search_resume instead - it
    will find more detail.

    Args:
        query: what to look for, in plain words
    """
    return emit(search_across_roles(query), tool_call_id)


@tool
def search_resume(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search the user's CV, interview preparation and work history.

    Their resumes, STAR answers, per-company interview prep, and notes on
    projects they actually worked on. Use for anything about *them*: what they
    did at a company, which projects they led, how they described a piece of
    work, what a job description asked for.

    Not for general technical reference: that is search_notes.

    Args:
        query: what to look for, in plain words
    """
    return emit(search(Collection.RESUME_INTERVIEW, query), tool_call_id)
