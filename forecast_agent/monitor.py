"""A standing watch over a legal area.

A forecast is a snapshot. A monitor is the same retrieval run on a schedule,
with each run diffed against the last one so you only hear about works that are
*newly* appearing. It deliberately skips the model: a diff of FRBR URIs needs no
LLM, and a monitor that costs nothing to run can run often.

    python -m forecast_agent.monitor --area "electricity generation licensing"

Run it from cron, or from any scheduler you like. State lives in a JSON file
keyed by (knowledge base, legal area).
"""

import argparse
import asyncio
import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import dotenv

from forecast_agent.graph import retrieve_law
from forecast_agent.kb import DEFAULT_KB, Work

STATE_FILE = Path("monitor_state.json")


async def check(legal_area: str, knowledge_base: str, state: dict) -> tuple[list[Work], list[Work], dict]:
    """Retrieve, diff against the stored baseline, and return the new works.

    The first run for an area establishes a baseline and reports nothing: with
    no prior run to compare against, every existing work would look "new".
    """
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


def diff(works: list[Work], previous_uris: list[str]) -> list[Work]:
    """Works whose FRBR URI wasn't in the previous run."""
    seen = set(previous_uris)
    return [w for w in works if w.frbr_uri not in seen]


def render_digest(legal_area: str, new_upcoming: list[Work], new_repealed: list[Work]) -> str:
    lines = [f"New activity in: {legal_area}", ""]
    if new_upcoming:
        lines.append("Upcoming:")
        lines += [f"  - {w.title} ({w.public_url})" for w in new_upcoming]
        lines.append("")
    if new_repealed:
        lines.append("Repealed:")
        lines += [f"  - {w.title} ({w.public_url})" for w in new_repealed]
        lines.append("")
    return "\n".join(lines)


def send_email(subject: str, body: str, to: list[str]) -> None:
    """Email the digest, if SMTP is configured. Otherwise this is a no-op."""
    host = os.environ.get("SMTP_HOST")
    if not host or not to:
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", "noreply@example.com")
    msg["To"] = ", ".join(to)
    msg.set_content(body)

    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as smtp:
        smtp.starttls()
        if os.environ.get("SMTP_USERNAME"):
            smtp.login(os.environ["SMTP_USERNAME"], os.environ.get("SMTP_PASSWORD", ""))
        smtp.send_message(msg)


def load_all_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


async def main(args) -> None:
    all_state = load_all_state(STATE_FILE)
    key = f"{args.knowledge_base}:{args.area}"

    new_upcoming, new_repealed, new_state = await check(
        args.area, args.knowledge_base, all_state.get(key, {})
    )

    if new_upcoming or new_repealed:
        digest = render_digest(args.area, new_upcoming, new_repealed)
        print(digest)
        send_email(f"Legislation update: {args.area}", digest, args.notify)
    else:
        print(f"No new works for: {args.area}")

    # Only persist the new baseline once the run has fully succeeded — a failed
    # run must not silently mark these works as "already seen".
    all_state[key] = new_state
    STATE_FILE.write_text(json.dumps(all_state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a legal area for new legislation.")
    parser.add_argument("--area", required=True, help="The legal area to watch")
    parser.add_argument("--knowledge-base", default=DEFAULT_KB)
    parser.add_argument(
        "--notify",
        action="append",
        default=[],
        help="Email address to notify (repeatable; requires SMTP_HOST)",
    )
    dotenv.load_dotenv()
    asyncio.run(main(parser.parse_args()))
