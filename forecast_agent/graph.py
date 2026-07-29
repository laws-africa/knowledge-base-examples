"""A legislation forecast agent.

Instead of answering a question from the law as it stands today, this agent
answers "what does the law say about X, and what is about to change?" It does
that with three filtered retrievals against the same knowledge base:

    current   commenced=True,  repealed=False, principal=True
    upcoming  commenced=False, repealed=False
    repealed  repealed=True

The three are independent, so they run concurrently. The model then writes a
narrative over the results, and every work it refers to is checked against what
was actually retrieved before it reaches the user.
"""

import asyncio
import re
from datetime import date, timedelta

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph
from pydantic import BaseModel, Field

from forecast_agent.kb import (
    DEFAULT_KB,
    Portion,
    Work,
    dedupe_portions,
    group_works,
    retrieve,
)

# Change this to use your desired model
MODEL = "openai/gpt-5-mini"

TOP_K = 10

# How far back a repeal still counts as "recent".
REPEAL_LOOKBACK_DAYS = 365

# How far back an uncommenced work's latest version may be dated and still be
# useful as a "what's changing soon" signal. `expression_date` is the date of
# that version, not a commencement date, so this is a heuristic, not a promise.
UNCOMMENCED_LOOKBACK_DAYS = 730


class ForecastState(MessagesState):
    """Graph state. `MessagesState` gives us the `messages` list for chat UIs."""

    legal_area: str
    knowledge_base: str
    current: list[Portion]
    upcoming: list[Work]
    repealed: list[Work]
    report: str
    generation_mode: str


class ChangeSummary(BaseModel):
    """The model's take on one work. `work_frbr_uri` identifies which one."""

    work_frbr_uri: str = Field(description="Copied exactly from the retrieved works")
    summary: str = Field(description="What this work does")
    impact: str = Field(description="Who it affects and how")


class Citation(BaseModel):
    work_frbr_uri: str
    portion_id: str


class ForecastOutput(BaseModel):
    """The structured answer we ask the model for."""

    current_law: str = Field(description="The present legal position, from Block A only")
    changes: list[ChangeSummary] = Field(default_factory=list)
    repeals: list[ChangeSummary] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a name in the format 'provider/model'."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


async def resolve_area(state: ForecastState):
    """Take the legal area from state, or from the last chat message."""
    legal_area = (state.get("legal_area") or "").strip()
    if not legal_area and state.get("messages"):
        legal_area = state["messages"][-1].content.strip()
    if not legal_area:
        raise ValueError("no legal area given")

    return {
        "legal_area": legal_area,
        "knowledge_base": state.get("knowledge_base") or DEFAULT_KB,
    }


async def retrieve_law(state: ForecastState):
    """Run the three retrievals concurrently, then group and filter the results."""
    kb = state["knowledge_base"]
    area = state["legal_area"]
    print(f"Retrieving current, upcoming and repealed law for: {area}")

    current, upcoming, repealed = await asyncio.gather(
        retrieve(kb, area, TOP_K, {"commenced": True, "repealed": False, "principal": True}),
        retrieve(kb, area, TOP_K, {"commenced": False, "repealed": False}),
        retrieve(kb, area, TOP_K, {"repealed": True}),
    )

    today = date.today()
    upcoming_works = group_works(
        upcoming, cutoff=today - timedelta(days=UNCOMMENCED_LOOKBACK_DAYS)
    )
    repealed_works = group_works(
        repealed, cutoff=today - timedelta(days=REPEAL_LOOKBACK_DAYS), newest_first=True
    )
    current_portions = dedupe_portions(current)

    print(
        f"Retrieved {len(current_portions)} current provisions, "
        f"{len(upcoming_works)} upcoming works, {len(repealed_works)} recent repeals"
    )
    return {
        "current": current_portions,
        "upcoming": upcoming_works,
        "repealed": repealed_works,
    }


async def synthesize(state: ForecastState):
    """Ask the model for a structured forecast, then verify it against retrieval.

    If retrieval found nothing, or the model call fails, we still return a
    usable report built from the retrieved metadata alone. A forecast without
    narrative is worth more than an error page.
    """
    current, upcoming, repealed = state["current"], state["upcoming"], state["repealed"]
    if not current and not upcoming and not repealed:
        report = f"No legislation found for **{state['legal_area']}** in `{state['knowledge_base']}`."
        return {"report": report, "generation_mode": "none", "messages": [AIMessage(report)]}

    try:
        llm = load_chat_model(MODEL).with_structured_output(ForecastOutput)
        print("Synthesizing forecast...")
        output = await llm.ainvoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(state)},
            ]
        )
        forecast = verify(output, state)
        mode = "llm"
    except Exception as e:  # noqa: BLE001 - any model failure degrades to heuristic
        print(f"Synthesis failed ({e}); falling back to retrieved results only.")
        forecast = None
        mode = "heuristic_fallback"

    report = render_report(state, forecast)
    return {"report": report, "generation_mode": mode, "messages": [AIMessage(report)]}


SYSTEM_PROMPT = """You are producing a legislation forecast: the current legal position for a legal area, \
and what is changing.

current_law summarises the present legal position from Block A only.
Every changes[].work_frbr_uri must be copied exactly from a work_frbr_uri in Block B — never invent one.
Every repeals[].work_frbr_uri must be copied exactly from a work_frbr_uri in Block C — never invent one.
repeals[] should note what the repealed work covered and what replaces it, if that is apparent from Block C.
Every citation must copy work_frbr_uri and portion_id exactly from a Block A provision.
Never state a legal source that is not in the blocks below.

In prose, never write a work_frbr_uri, portion id, or any /akn/... path — those belong only in the \
structured fields. Refer to legislation by name and section number, e.g. "section 34 of the Electricity \
Regulation Act, 2006"."""


def build_prompt(state: ForecastState) -> str:
    """Lay the three retrievals out as labelled blocks the prompt can point at."""
    blocks = [
        f"Legal area: {state['legal_area']}",
        "",
        "Block A — current law (commenced, not repealed):",
        format_portions(state["current"]) or "(none retrieved)",
        "",
        "Block B — pending legislation (not yet commenced), grouped by work:",
        format_works(state["upcoming"]) or "(none retrieved)",
        "",
        "Block C — recently repealed legislation, grouped by work:",
        format_works(state["repealed"], date_label="repeal_date") or "(none retrieved)",
    ]
    return "\n".join(blocks)


def format_portions(portions: list[Portion]) -> str:
    lines = []
    for i, p in enumerate(portions, start=1):
        heading = f" — {p.portion_title}" if p.portion_title else ""
        lines.append(
            f"[{i}] {p.title}{heading} "
            f"(work_frbr_uri: {p.work_frbr_uri}, portion_id: {p.portion_id})"
        )
        lines.append(clip(p.text, 1400))
        lines.append("")
    return "\n".join(lines).strip()


def format_works(works: list[Work], date_label: str = "expression_date") -> str:
    lines = []
    for i, w in enumerate(works, start=1):
        dated = f", {date_label}: {w.expression_date}" if w.expression_date else ""
        lines.append(f"[{i}] {w.title} (work_frbr_uri: {w.frbr_uri}{dated})")
        for p in w.portions:
            prefix = f"{p.portion_title}: " if p.portion_title else ""
            lines.append(f"  - {prefix}{clip(p.text, 500)}")
        lines.append("")
    return "\n".join(lines).strip()


def clip(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].strip() + "..."


def verify(output: ForecastOutput, state: ForecastState) -> ForecastOutput:
    """Drop anything the model said that retrieval doesn't support.

    Structured output guarantees the *shape* of the answer, not its truth: the
    model can still return a plausible-looking FRBR URI for a work that was
    never retrieved. Anything that doesn't match is dropped rather than shown.
    """
    upcoming_uris = {w.frbr_uri for w in state["upcoming"]}
    repealed_uris = {w.frbr_uri for w in state["repealed"]}
    current_chunks = {(p.work_frbr_uri, p.portion_id) for p in state["current"]}

    return ForecastOutput(
        current_law=strip_frbr_artifacts(output.current_law),
        changes=[
            clean_change(c) for c in output.changes if c.work_frbr_uri in upcoming_uris
        ],
        repeals=[
            clean_change(c) for c in output.repeals if c.work_frbr_uri in repealed_uris
        ],
        citations=[
            c for c in output.citations if (c.work_frbr_uri, c.portion_id) in current_chunks
        ],
    )


def clean_change(change: ChangeSummary) -> ChangeSummary:
    return ChangeSummary(
        work_frbr_uri=change.work_frbr_uri,
        summary=strip_frbr_artifacts(change.summary),
        impact=strip_frbr_artifacts(change.impact),
    )


FRBR_PAREN = re.compile(r"\([^()]*work_frbr_uri[^()]*\)")
FRBR_PATH = re.compile(r"/akn/[^\s,()]+")
EMPTY_PAREN = re.compile(r"\(\s*\)")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
REPEATED_SPACES = re.compile(r"[ \t]{2,}")


def strip_frbr_artifacts(text: str) -> str:
    """Remove machine identifiers that leak into prose despite the instruction.

    Models are good at following this rule and not perfect at it. The URIs are
    already carried in structured fields, so a reader should never see one in a
    sentence — this is a cheap, generic backstop rather than a per-model fix.
    """
    text = FRBR_PAREN.sub("", text)
    text = FRBR_PATH.sub("", text)
    text = EMPTY_PAREN.sub("", text)
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = REPEATED_SPACES.sub(" ", text)
    return text.strip()


def render_report(state: ForecastState, forecast: ForecastOutput | None) -> str:
    """Render the forecast as markdown.

    `forecast` is None when synthesis failed — the retrieved works are still
    listed, just without narrative.
    """
    summaries = {c.work_frbr_uri: c for c in forecast.changes} if forecast else {}
    repeal_summaries = {c.work_frbr_uri: c for c in forecast.repeals} if forecast else {}

    lines = [f"# Legislation forecast: {state['legal_area']}", ""]

    lines.append("## Current law")
    if forecast:
        lines += [forecast.current_law, ""]
    else:
        lines += ["_AI analysis unavailable; retrieved provisions are listed below._", ""]
        for p in state["current"][:5]:
            heading = f" — {p.portion_title}" if p.portion_title else ""
            lines.append(f"- [{p.title}{heading}]({p.portion_public_url or p.public_url})")
        lines.append("")

    lines.append("## Upcoming changes")
    lines += render_works(state["upcoming"], summaries, "expected") or ["_None found._", ""]

    lines.append("## Recent repeals")
    lines += render_works(state["repealed"], repeal_summaries, "repealed") or ["_None found._", ""]

    if forecast and forecast.citations:
        lines.append("## Citations")
        by_chunk = {(p.work_frbr_uri, p.portion_id): p for p in state["current"]}
        for c in forecast.citations:
            p = by_chunk[(c.work_frbr_uri, c.portion_id)]
            heading = p.portion_title or p.portion_id
            lines.append(f"- [{p.title} — {heading}]({p.portion_public_url or p.public_url})")
        lines.append("")

    lines.append(
        f"_Retrieved from the Laws.Africa `{state['knowledge_base']}` knowledge base, "
        "which is in preview and may be incomplete or not fully up to date._"
    )
    return "\n".join(lines)


def render_works(
    works: list[Work], summaries: dict[str, ChangeSummary], date_word: str
) -> list[str]:
    lines: list[str] = []
    for w in works:
        dated = f" — {date_word} {w.expression_date}" if w.expression_date else ""
        lines.append(f"### [{w.title}]({w.public_url}){dated}")
        summary = summaries.get(w.frbr_uri)
        if summary:
            lines += [summary.summary, "", f"**Impact:** {summary.impact}"]
        lines.append("")
    return lines


# Define the forecast graph: resolve the area, retrieve, then synthesize.
builder = StateGraph(ForecastState)

builder.add_node("resolve_area", resolve_area)
builder.add_node("retrieve", retrieve_law)
builder.add_node("synthesize", synthesize)
builder.add_edge("__start__", "resolve_area")
builder.add_edge("resolve_area", "retrieve")
builder.add_edge("retrieve", "synthesize")

forecast_graph = builder.compile(name="Legislation Forecast Agent")
