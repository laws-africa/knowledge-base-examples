# Tutorial: build a legislation forecast agent

The other examples in this repo answer questions about the law *as it stands*.
This one answers a different question:

> What does the law say about X right now, and what is about to change?

That question comes up constantly in compliance work — a company doesn't only
need to know the rule, it needs to know whether the rule is about to move. This
tutorial builds an agent that answers it, using nothing but the Laws.Africa
[Knowledge Base API](https://developers.laws.africa/ai-api/knowledge-bases), an
LLM, and about 400 lines of Python.

The finished code is in [`forecast_agent/`](forecast_agent/). Read this
alongside it, or build it up file by file as you go.

Here is what it produces:

```
# Legislation forecast: electricity generation licensing

## Current law
A licence from the National Energy Regulator is required to operate a generation,
transmission or distribution facility, under section 7 of the Electricity Regulation
Act, 2006. Section 8 exempts certain activities...

## Upcoming changes
### Electricity Regulation Amendment Act — expected 2026-02-06
Widens the exemption for small-scale embedded generation.

**Impact:** Generators below the threshold no longer need a licence, only registration.

## Recent repeals
### Old Licensing Notice — repealed 2025-11-01
...

## Citations
- Electricity Regulation Act, 2006 — 7. Activities requiring licensing
```

---

## The idea: one question, three retrievals

The naive approach is to search the knowledge base for "electricity generation
licensing" and ask a model what's changing. That doesn't work, because the
answer to "what's changing" isn't in the text of the current law — it's in a
*different set of documents* that the current law never mentions.

The Knowledge Base API has the filters we need to go get them:

| Retrieval | Filters | Answers |
|---|---|---|
| Current law | `commenced=True, repealed=False, principal=True` | What is the rule today? |
| Upcoming | `commenced=False, repealed=False` | What has been passed but isn't in force yet? |
| Repealed | `repealed=True` | What has recently fallen away? |

Same query text, same knowledge base, three different windows onto it. That
table *is* the feature. Everything else in this tutorial is plumbing around it:
turning chunks into documents, keeping the model honest, and rendering a report.

## Before you start

You need:

1. Python 3.11 or later
2. An OpenAI API key
3. A Laws.Africa API key ([how to get one](https://developers.laws.africa/ai-api/authentication))

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put your keys in a `.env` file at the repo root:

```bash
OPENAI_API_KEY='your-openai-api-key'
LAWSAFRICA_API_TOKEN='your-laws-africa-api-key'
```

---

## Step 1 — Talk to the Knowledge Base

The retrieve endpoint takes a query, a `top_k`, and optional filters, and returns
scored chunks. It does retrieval only — no generation. What we do with the
results is entirely up to us.

```python
# forecast_agent/kb.py
KB_API_URL = "https://api.laws.africa/ai/v1/knowledge-bases"
DEFAULT_KB = "legislation-za"


async def retrieve(kb_name, text, top_k=10, filters=None):
    token = os.environ.get("LAWSAFRICA_API_TOKEN")
    if not token:
        raise RuntimeError("LAWSAFRICA_API_TOKEN is not set")

    payload = {"text": text, "top_k": top_k}
    if filters:
        payload["filters"] = filters

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{KB_API_URL}/{kb_name}/retrieve",
            headers={"Authorization": f"Token {token}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

    return [Portion.from_result(r) for r in data.get("results", [])]
```

Each result has `content.text`, a `score`, and a `metadata` block. Rather than
passing raw dicts around, we parse them into a `Portion` model straight away —
so a typo in a metadata key fails here, once, instead of silently producing an
empty string in a prompt three functions later.

```python
class Portion(BaseModel):
    """One scored chunk returned by the Knowledge Base API.

    `score` is a distance — *lower* is better. Don't mix it with cosine
    similarity scores from another retriever without normalising first.
    """

    text: str
    score: float
    work_frbr_uri: str
    title: str
    expression_date: str = ""
    public_url: str = ""
    portion_id: str = ""
    portion_title: str = ""
    portion_public_url: str = ""
    portion_parent_ids: list[str] = Field(default_factory=list)
```

Two fields carry most of the weight for the rest of this tutorial:

- **`work_frbr_uri`** — the stable identifier for a piece of legislation, e.g.
  `/akn/za/act/2006/4`. Different versions of the same Act share it. It is how
  we group chunks, how we diff runs, and how we catch a model inventing a source.
- **`expression_date`** — the date of *this version* of the document. Note what
  it is not: it is not a commencement date. More on that in Step 2.

## Step 2 — From chunks to documents

The API returns portions. A forecast is about works: "the Amendment Act is
coming", not "chunk `sec_5` is coming". So we group.

```python
def group_works(portions, cutoff=None, newest_first=False):
    works = {}
    for p in portions:
        if not p.work_frbr_uri:
            continue
        work = works.get(p.work_frbr_uri)
        if work is None:
            work = Work(
                frbr_uri=p.work_frbr_uri,
                title=p.title,
                expression_date=p.expression_date,
                public_url=p.public_url,
            )
            works[p.work_frbr_uri] = work
        work.portions.append(p)

    result = list(works.values())
    if cutoff is not None:
        result = [
            w
            for w in result
            if (parsed := parse_date(w.expression_date)) is not None and parsed >= cutoff
        ]
    result.sort(key=lambda w: w.expression_date, reverse=newest_first)
    return result
```

Two judgement calls are baked in here, and both are worth understanding before
you copy this.

**The cutoff.** An uncommenced work whose latest version is dated 2015 is not
news; it's a bill that stalled a decade ago. We drop it. The example uses two
years for uncommenced works and one year for repeals:

```python
REPEAL_LOOKBACK_DAYS = 365
UNCOMMENCED_LOOKBACK_DAYS = 730
```

This is a heuristic standing in for a commencement date that the retrieve
endpoint doesn't expose. It will occasionally drop something you wanted and
occasionally keep something stale. Tune it, and be honest with your users about
what the dates mean.

**Undated works are dropped, not kept.** If `expression_date` is missing or
unparseable, we can't prove the work is recent — so we don't show it as recent.
When a filter is about recency, failing closed is the safer default.

There's one more cleanup. The knowledge base indexes overlapping granularities,
so a query can match both a chapter and a section inside that chapter. Keeping
both spends your prompt budget twice on the same text, and gives the reader two
citations that point at the same rule:

```python
def dedupe_portions(portions):
    seen = set()
    unique = []
    for p in portions:
        key = (p.work_frbr_uri, p.portion_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    ancestors = {
        (p.work_frbr_uri, parent_id)
        for p in unique
        for parent_id in p.portion_parent_ids
    }
    return [p for p in unique if (p.work_frbr_uri, p.portion_id) not in ancestors]
```

`portion_parent_ids` gives us the ancestry, so when we hold a section we can
drop the chapter that contains it.

## Step 3 — Retrieve three ways, at once

Now the core. The three retrievals don't depend on each other, so running them
in sequence would triple the latency for no reason. `asyncio.gather` runs them
concurrently, and the node finishes as fast as the slowest one:

```python
# forecast_agent/graph.py
async def retrieve_law(state):
    kb = state["knowledge_base"]
    area = state["legal_area"]

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

    return {
        "current": current_portions,
        "upcoming": upcoming_works,
        "repealed": repealed_works,
    }
```

Note the sort directions. Upcoming works go **oldest first** — the most imminent
change is the one that's been waiting longest. Repeals go **newest first** — the
most recent repeal is the most relevant. Same helper, opposite `newest_first`.

`principal=True` on the current-law retrieval keeps it to principal Acts rather
than the amendment notices that modify them. Amendments matter for the *upcoming*
side, which is why that filter isn't repeated there.

This is a LangGraph node, so the state it reads and writes is declared up front:

```python
class ForecastState(MessagesState):
    legal_area: str
    knowledge_base: str
    current: list[Portion]
    upcoming: list[Work]
    repealed: list[Work]
    report: str
    generation_mode: str
```

Subclassing `MessagesState` gets us the `messages` list, which is what lets this
agent work in a chat UI alongside the other two examples.

## Step 4 — Ask the model, in blocks

The model now has three distinct piles of text to work with, and the answer
depends on it keeping them apart — the current position comes from one pile, and
"what's changing" from another. So label them explicitly in the prompt:

```python
def build_prompt(state):
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
```

Named blocks give the system prompt something concrete to point at:

```
current_law summarises the present legal position from Block A only.
Every changes[].work_frbr_uri must be copied exactly from a work_frbr_uri in Block B — never invent one.
Every repeals[].work_frbr_uri must be copied exactly from a work_frbr_uri in Block C — never invent one.
Every citation must copy work_frbr_uri and portion_id exactly from a Block A provision.
Never state a legal source that is not in the blocks below.
```

We ask for structured output rather than prose, because we want to render the
result ourselves and — more importantly — check it:

```python
class ChangeSummary(BaseModel):
    work_frbr_uri: str = Field(description="Copied exactly from the retrieved works")
    summary: str = Field(description="What this work does")
    impact: str = Field(description="Who it affects and how")


class ForecastOutput(BaseModel):
    current_law: str = Field(description="The present legal position, from Block A only")
    changes: list[ChangeSummary] = Field(default_factory=list)
    repeals: list[ChangeSummary] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
```

```python
llm = load_chat_model(MODEL).with_structured_output(ForecastOutput)
output = await llm.ainvoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_prompt(state)},
])
```

Each work is identified by its FRBR URI rather than by title or position, so the
next step can look it up unambiguously.

## Step 5 — Verify what the model said

This is the step people skip, and it's the one that matters most in a legal
tool.

Structured output guarantees the *shape* of an answer, not its truth. A model
that has read a hundred FRBR URIs can produce a hundred-and-first that looks
perfectly well-formed and refers to an Act that was never retrieved — or worse,
never existed. So every identifier the model returns is checked against what
retrieval actually found, and anything that doesn't match is dropped:

```python
def verify(output, state):
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
```

Dropped, not flagged. A citation the reader can't click through and check is
worse than no citation at all.

Notice too that the title, date and URL shown in the final report are taken from
the retrieved `Work`, never from the model. The model's job is analysis; the
facts come from the API.

The second half of verification is cosmetic but visible. Despite being told not
to, models will sometimes write a URI into prose — `"licences are required
(work_frbr_uri: /akn/za/act/2006/4)"`. The identifiers are already in the
structured fields, so we strip them from the narrative:

```python
FRBR_PAREN = re.compile(r"\([^()]*work_frbr_uri[^()]*\)")
FRBR_PATH = re.compile(r"/akn/[^\s,()]+")


def strip_frbr_artifacts(text):
    text = FRBR_PAREN.sub("", text)
    text = FRBR_PATH.sub("", text)
    text = EMPTY_PAREN.sub("", text)
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = REPEATED_SPACES.sub(" ", text)
    return text.strip()
```

Keep backstops like this generic — a pattern for "anything FRBR-shaped", not a
rule about one Act or one phrasing. If you find yourself special-casing a
particular query's bad output, the fix almost always belongs further upstream in
retrieval or the prompt.

## Step 6 — Degrade to something useful

Model calls fail: rate limits, timeouts, outages. The retrieval already
succeeded at that point, and a list of upcoming Acts with dates and links is
genuinely useful even with no narrative over it. So a synthesis failure falls
back rather than erroring out:

```python
    try:
        llm = load_chat_model(MODEL).with_structured_output(ForecastOutput)
        output = await llm.ainvoke([...])
        forecast = verify(output, state)
        mode = "llm"
    except Exception as e:
        print(f"Synthesis failed ({e}); falling back to retrieved results only.")
        forecast = None
        mode = "heuristic_fallback"

    report = render_report(state, forecast)
```

`render_report` takes `forecast=None` and simply renders the retrieved works
without summaries. The user gets the works, plus a line telling them the
analysis is missing — and `generation_mode` in the state says which path ran, so
a caller can surface that too. Failing loudly to the operator and softly to the
reader is the right combination here.

There's a third mode: retrieval that finds nothing at all. Say so plainly rather
than inviting the model to fill the silence:

```python
if not current and not upcoming and not repealed:
    report = f"No legislation found for **{state['legal_area']}** in `{state['knowledge_base']}`."
    return {"report": report, "generation_mode": "none", ...}
```

## Step 7 — Wire up the graph

Three nodes, in a line:

```python
builder = StateGraph(ForecastState)

builder.add_node("resolve_area", resolve_area)
builder.add_node("retrieve", retrieve_law)
builder.add_node("synthesize", synthesize)
builder.add_edge("__start__", "resolve_area")
builder.add_edge("resolve_area", "retrieve")
builder.add_edge("retrieve", "synthesize")

forecast_graph = builder.compile(name="Legislation Forecast Agent")
```

`resolve_area` exists so the agent can be driven two ways: programmatically with
`{"legal_area": "..."}`, or from a chat UI where the area arrives as a message.

```python
async def resolve_area(state):
    legal_area = (state.get("legal_area") or "").strip()
    if not legal_area and state.get("messages"):
        legal_area = state["messages"][-1].content.strip()
    if not legal_area:
        raise ValueError("no legal area given")

    return {
        "legal_area": legal_area,
        "knowledge_base": state.get("knowledge_base") or DEFAULT_KB,
    }
```

Register it in `langgraph.json` next to the other agents:

```json
{
  "graphs": {
    "legislation_agent": "./kb_agent/graph.py:legislation_graph",
    "judgment_agent": "./kb_agent/graph.py:judgment_graph",
    "forecast_agent": "./forecast_agent/graph.py:forecast_graph"
  }
}
```

Then run it:

```bash
python agent.py forecast
# then enter e.g. "electricity generation licensing"
```

or in the visual UI:

```bash
langgraph dev --no-browser
# open https://agentchat.vercel.app/?apiUrl=http://localhost:2024 and enter forecast_agent
```

## Step 8 — From a forecast to a monitor

A forecast is a snapshot: you ask, you get today's picture. The more valuable
product is the standing version — *tell me when something changes in this area*.

That turns out to be a small addition, because it's the same retrieval run twice
and compared. Store the FRBR URIs from each run, and the next run reports only
what wasn't there before:

```python
# forecast_agent/monitor.py
def diff(works, previous_uris):
    seen = set(previous_uris)
    return [w for w in works if w.frbr_uri not in seen]
```

The whole check reuses the graph's retrieval node directly — nodes are just
functions over a state dict, so there's nothing stopping you calling one outside
a graph:

```python
async def check(legal_area, knowledge_base, state):
    result = await retrieve_law({"legal_area": legal_area, "knowledge_base": knowledge_base})
    upcoming, repealed = result["upcoming"], result["repealed"]

    first_run = "last_run_at" not in state
    new_upcoming = [] if first_run else diff(upcoming, state.get("upcoming_uris", []))
    new_repealed = [] if first_run else diff(repealed, state.get("repealed_uris", []))

    new_state = {
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "upcoming_uris": [w.frbr_uri for w in upcoming],
        "repealed_uris": [w.frbr_uri for w in repealed],
    }
    return new_upcoming, new_repealed, new_state
```

Three details are doing real work here:

- **No model call.** Diffing a set of URIs doesn't need an LLM, and a monitor
  that costs nothing to run can run daily without anyone thinking about the
  bill. Save synthesis for when a human actually opens the report.
- **The first run is silent.** With no previous run to compare against, every
  existing work would look new and the first email would be a wall of noise.
  The first run establishes the baseline and reports nothing.
- **The baseline is only saved after a successful run.** If retrieval fails, the
  old baseline stays, and the next run picks up everything that was missed. A
  failed run must never silently mark works as "already seen".

```bash
python -m forecast_agent.monitor --area "electricity generation licensing"
```

State goes into `monitor_state.json`, keyed by knowledge base and area, so one
file can watch many areas. Point cron at it and it's a service:

```cron
0 7 * * * cd /path/to/repo && .venv/bin/python -m forecast_agent.monitor --area "electricity generation licensing" --notify you@example.com
```

`--notify` sends the digest by email when `SMTP_HOST` is configured; without it
the digest just prints.

## Testing it without API keys

Both external dependencies are injected at a single named function, so both can
be faked. That makes the interesting cases — a hallucinated citation, a model
outage, a stale work — cheap to test, and testable in CI with no secrets:

```python
@pytest.fixture
def fake_kb(monkeypatch):
    async def fake_retrieve(kb_name, text, top_k=10, filters=None):
        filters = filters or {}
        if filters.get("repealed"):
            return [portion("/akn/za/act/2020/2", "sec_1", title="Old Act")]
        if filters.get("commenced"):
            return [portion("/akn/za/act/2006/4", "sec_34", title="Electricity Regulation Act")]
        return [portion("/akn/za/act/2026/1", "sec_5", title="Amendment Act")]

    monkeypatch.setattr(fg, "retrieve", fake_retrieve)
```

Keying the fake on `filters` is what lets one fixture serve all three
retrievals — and it doubles as a check that the node is sending the filters you
think it is.

The test worth writing first is the one that proves verification works:

```python
@pytest.mark.asyncio
async def test_forecast_drops_works_and_citations_that_were_never_retrieved(fake_kb, monkeypatch):
    fake_model(monkeypatch, fg.ForecastOutput(
        current_law="Licences are required.",
        changes=[fg.ChangeSummary(
            work_frbr_uri="/akn/za/act/2027/99",   # never retrieved
            summary="Invented act.",
            impact="Invented impact.",
        )],
        citations=[fg.Citation(work_frbr_uri="/akn/za/act/2006/4", portion_id="sec_999")],
    ))

    state = await run_graph()

    assert "Invented act." not in state["report"]
    assert "## Citations" not in state["report"]
```

The full suite is in [`test_forecast_agent.py`](test_forecast_agent.py):

```bash
pip install -r requirements-dev.txt
pytest
```

## What to watch out for

Things that will bite you when you point this at a real corpus:

- **Semantic retrieval returns near misses.** Querying "electricity generation
  licensing" against `legislation-za` will surface the Electricity Regulation
  Act — and, a few results down, an unrelated shipping regulation that happens
  to talk about licences and power. Raise `top_k` and you get more of both. If
  precision matters more than recall for your use case, narrow with
  `frbr_place`, `frbr_doctype` or `frbr_subtype` filters rather than by
  post-filtering the text.
- **`expression_date` is not a commencement date.** It's the date of that
  version of the document. An uncommenced work's expression date tells you when
  the text was published, not when it takes effect. Don't render it as "in force
  from" — this example says "expected", which is already generous.
- **`score` is a distance, so lower is better.** The opposite convention to
  cosine similarity. If you merge these results with another retriever's, or
  sort by score, check which way round you are.
- **The knowledge bases are in preview.** Coverage is incomplete and not always
  current. Say so in your output — the example appends a caveat line to every
  report — and don't let a forecast be the last word on whether something is in
  force.
- **`top_k` applies per retrieval, before grouping.** Ten portions might be ten
  works or two. If you consistently see too few upcoming works, that's usually
  the cause.

## Where to take it next

- **More jurisdictions.** Every filter here is jurisdiction-neutral; only the
  knowledge base name and any `frbr_place` filter are local. Pass
  `knowledge_base` through the state and the same graph covers a different
  country.
- **Persist the runs.** Swapping `monitor_state.json` for a database gets you
  history: which works appeared when, and what a monitor has already told
  someone about.
- **Stream the report.** Retrieval takes well under a second; synthesis takes
  tens of seconds. Emitting the retrieved works as soon as they're grouped,
  then filling in the narrative when it arrives, makes the wait far less
  noticeable than a spinner does.
- **Watch specific works, not just areas.** Filter with `work_frbr_uri__in` and
  the same diff logic becomes "tell me when *this Act* changes".
