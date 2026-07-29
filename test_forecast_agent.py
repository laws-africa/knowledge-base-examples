"""Tests for the forecast agent, with no API keys and no network calls.

Both external services — the Knowledge Base and the model — are replaced with
fakes, which is what makes the interesting cases (a hallucinated citation, a
model outage, a stale work) cheap to test at all.

    pip install -r requirements-dev.txt
    pytest
"""

from datetime import date, timedelta

import pytest

from forecast_agent import graph as fg
from forecast_agent import monitor as fm
from forecast_agent.kb import Portion, Work, dedupe_portions, group_works

TODAY = date.today()
RECENT = (TODAY - timedelta(days=30)).isoformat()
STALE = (TODAY - timedelta(days=3000)).isoformat()


def portion(work_uri, portion_id, *, parents=(), title="An Act", date_=RECENT, text="text"):
    return Portion(
        text=text,
        score=0.1,
        work_frbr_uri=work_uri,
        title=title,
        expression_date=date_,
        public_url=f"https://example.org{work_uri}",
        portion_id=portion_id,
        portion_title=portion_id,
        portion_public_url=f"https://example.org{work_uri}#{portion_id}",
        portion_parent_ids=list(parents),
    )


# ── kb helpers ────────────────────────────────────────────────────────────────


def test_dedupe_drops_exact_duplicates_and_ancestors():
    chapter = portion("/akn/za/act/2006/4", "chp_2")
    section = portion("/akn/za/act/2006/4", "sec_34", parents=["chp_2"])
    other_act_chapter = portion("/akn/za/act/2010/9", "chp_2")

    result = dedupe_portions([chapter, section, chapter, other_act_chapter])

    ids = [(p.work_frbr_uri, p.portion_id) for p in result]
    assert ids == [("/akn/za/act/2006/4", "sec_34"), ("/akn/za/act/2010/9", "chp_2")]


def test_group_works_drops_stale_and_undated_works():
    portions = [
        portion("/akn/za/act/2026/1", "sec_1", date_=RECENT),
        portion("/akn/za/act/2026/1", "sec_2", date_=RECENT),
        portion("/akn/za/act/1998/7", "sec_1", date_=STALE),
        portion("/akn/za/act/2026/9", "sec_1", date_=""),
    ]

    works = group_works(portions, cutoff=TODAY - timedelta(days=365))

    assert [w.frbr_uri for w in works] == ["/akn/za/act/2026/1"]
    assert len(works[0].portions) == 2


def test_group_works_sort_order():
    portions = [
        portion("/akn/za/act/a", "sec_1", date_="2026-03-01"),
        portion("/akn/za/act/b", "sec_1", date_="2026-01-01"),
    ]

    assert [w.frbr_uri for w in group_works(portions)] == ["/akn/za/act/b", "/akn/za/act/a"]
    assert [w.frbr_uri for w in group_works(portions, newest_first=True)] == [
        "/akn/za/act/a",
        "/akn/za/act/b",
    ]


# ── the graph ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_kb(monkeypatch):
    """Return canned results per retrieval, keyed by the filters used."""

    async def fake_retrieve(kb_name, text, top_k=10, filters=None):
        filters = filters or {}
        if filters.get("repealed"):
            return [portion("/akn/za/act/2020/2", "sec_1", title="Old Act")]
        if filters.get("commenced"):
            return [portion("/akn/za/act/2006/4", "sec_34", title="Electricity Regulation Act")]
        return [portion("/akn/za/act/2026/1", "sec_5", title="Amendment Act")]

    monkeypatch.setattr(fg, "retrieve", fake_retrieve)


def fake_model(monkeypatch, output):
    """Replace the chat model with one that always returns `output`."""

    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            if isinstance(output, Exception):
                raise output
            return output

    monkeypatch.setattr(fg, "load_chat_model", lambda _name: FakeModel())


async def run_graph(area="electricity"):
    return await fg.forecast_graph.ainvoke({"legal_area": area})


@pytest.mark.asyncio
async def test_forecast_includes_verified_changes(fake_kb, monkeypatch):
    fake_model(
        monkeypatch,
        fg.ForecastOutput(
            current_law="Generation licences are required under section 34.",
            changes=[
                fg.ChangeSummary(
                    work_frbr_uri="/akn/za/act/2026/1",
                    summary="Widens the exemption.",
                    impact="Small generators.",
                )
            ],
            repeals=[
                fg.ChangeSummary(
                    work_frbr_uri="/akn/za/act/2020/2",
                    summary="Removed the old licensing regime.",
                    impact="Replaced by the 2026 Act.",
                )
            ],
            citations=[
                fg.Citation(work_frbr_uri="/akn/za/act/2006/4", portion_id="sec_34")
            ],
        ),
    )

    state = await run_graph()

    assert state["generation_mode"] == "llm"
    assert "Widens the exemption." in state["report"]
    assert "Removed the old licensing regime." in state["report"]
    assert "Electricity Regulation Act" in state["report"]


@pytest.mark.asyncio
async def test_forecast_drops_works_and_citations_that_were_never_retrieved(
    fake_kb, monkeypatch
):
    fake_model(
        monkeypatch,
        fg.ForecastOutput(
            current_law="Licences are required.",
            changes=[
                fg.ChangeSummary(
                    work_frbr_uri="/akn/za/act/2027/99",
                    summary="Invented act.",
                    impact="Invented impact.",
                )
            ],
            repeals=[],
            citations=[
                fg.Citation(work_frbr_uri="/akn/za/act/2006/4", portion_id="sec_999")
            ],
        ),
    )

    state = await run_graph()

    assert "Invented act." not in state["report"]
    assert "## Citations" not in state["report"]


@pytest.mark.asyncio
async def test_forecast_falls_back_when_the_model_fails(fake_kb, monkeypatch):
    fake_model(monkeypatch, RuntimeError("model unavailable"))

    state = await run_graph()

    assert state["generation_mode"] == "heuristic_fallback"
    # The retrieved works still make it to the reader.
    assert "Amendment Act" in state["report"]
    assert "AI analysis unavailable" in state["report"]


@pytest.mark.asyncio
async def test_forecast_reports_nothing_found(monkeypatch):
    async def empty_retrieve(*_args, **_kwargs):
        return []

    monkeypatch.setattr(fg, "retrieve", empty_retrieve)

    state = await run_graph(area="maritime salvage")

    assert state["generation_mode"] == "none"
    assert "No legislation found" in state["report"]


def test_strip_frbr_artifacts_removes_machine_identifiers():
    text = "See section 34 (work_frbr_uri: /akn/za/act/2006/4) and /akn/za/act/2026/1 too ."

    assert fg.strip_frbr_artifacts(text) == "See section 34 and too."


# ── the monitor ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_monitor_first_run_only_sets_a_baseline(fake_kb):
    new_upcoming, new_repealed, state = await fm.check("electricity", "legislation-za", {})

    assert new_upcoming == [] and new_repealed == []
    assert state["upcoming_uris"] == ["/akn/za/act/2026/1"]
    assert state["repealed_uris"] == ["/akn/za/act/2020/2"]


@pytest.mark.asyncio
async def test_monitor_reports_only_works_new_since_the_last_run(fake_kb):
    baseline = {
        "last_run_at": "2026-01-01T00:00:00+00:00",
        "upcoming_uris": [],
        "repealed_uris": ["/akn/za/act/2020/2"],
    }

    new_upcoming, new_repealed, _ = await fm.check("electricity", "legislation-za", baseline)

    assert [w.frbr_uri for w in new_upcoming] == ["/akn/za/act/2026/1"]
    assert new_repealed == []


def test_monitor_diff_ignores_known_uris():
    works = [Work(frbr_uri="/akn/a", title="A"), Work(frbr_uri="/akn/b", title="B")]

    assert [w.frbr_uri for w in fm.diff(works, ["/akn/a"])] == ["/akn/b"]
