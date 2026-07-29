"""Knowledge Base retrieval and work-grouping helpers for the forecast agent.

The Knowledge Base API returns *portions* (chunks of a document), not documents.
A forecast needs to reason about whole works — "the Electricity Regulation
Amendment Act is coming" — so everything here is about turning a flat list of
scored portions into a deduplicated, date-filtered list of works.
"""

import os
from datetime import date, datetime

import httpx
from pydantic import BaseModel, Field

KB_API_URL = "https://api.laws.africa/ai/v1/knowledge-bases"

# Default knowledge base. `legislation-za` covers South African national and
# provincial legislation; `legislation-za-municipal` covers by-laws.
DEFAULT_KB = "legislation-za"


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

    @classmethod
    def from_result(cls, result: dict) -> "Portion":
        meta = result.get("metadata", {})
        return cls(
            text=result.get("content", {}).get("text", ""),
            score=result.get("score", 0.0),
            work_frbr_uri=meta.get("work_frbr_uri", ""),
            title=meta.get("title", ""),
            expression_date=meta.get("expression_date") or "",
            public_url=meta.get("public_url") or "",
            portion_id=meta.get("portion_id") or "",
            portion_title=meta.get("portion_title") or "",
            portion_public_url=meta.get("portion_public_url") or "",
            portion_parent_ids=meta.get("portion_parent_ids") or [],
        )


class Work(BaseModel):
    """A single piece of legislation, with the portions that matched the query."""

    frbr_uri: str
    title: str
    expression_date: str = ""
    public_url: str = ""
    portions: list[Portion] = Field(default_factory=list)


async def retrieve(
    kb_name: str,
    text: str,
    top_k: int = 10,
    filters: dict | None = None,
) -> list[Portion]:
    """Semantic search against a knowledge base. Returns scored portions.

    This endpoint does retrieval only — no generation. Prompting a model with
    the results is our job.
    """
    token = os.environ.get("LAWSAFRICA_API_TOKEN")
    if not token:
        raise RuntimeError("LAWSAFRICA_API_TOKEN is not set")

    payload: dict = {"text": text, "top_k": top_k}
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


def dedupe_portions(portions: list[Portion]) -> list[Portion]:
    """Drop duplicate portions, and drop ancestors of portions we already have.

    The knowledge base indexes overlapping granularities, so a query can match
    both a whole chapter and a section inside that chapter. Keeping both spends
    the prompt budget twice on the same text, so when a descendant is present
    the ancestor is suppressed.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Portion] = []
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


def group_works(
    portions: list[Portion],
    cutoff: date | None = None,
    newest_first: bool = False,
) -> list[Work]:
    """Group portions into works, filter by date, and sort by date.

    `cutoff` drops works whose latest known version predates it. Works with a
    missing or unparseable `expression_date` are also dropped when a cutoff is
    given: we can't prove they're recent, so we don't claim they are.
    """
    works: dict[str, Work] = {}
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


def parse_date(value: str) -> date | None:
    """Parse an ISO `YYYY-MM-DD` expression date, or None if it's unusable."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
