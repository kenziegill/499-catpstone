"""Readout CLI — main entry point for ingesting reports and asking questions.

Two commands:
  readout ingest <pdf>   - extract findings from a PDF and store in the DB
  readout ask <question> - run the Research Analyst agent on a question

The 'ask' command resolves finding IDs back to verbatim quotes from the
database. The agent never outputs quote text directly — this is the
structural guarantee against quote hallucination.
"""
import re
import sys
from typing import Optional

import typer
from dotenv import load_dotenv

from agent import run_agent
from db import get_conn
from ingest import ingest as run_ingest

load_dotenv()

app = typer.Typer(
    help="Readout: a UX research analyst agent that reasons across a research corpus.",
    no_args_is_help=True,
)


# Regex to extract finding ID citations from the agent's answer.
# Format: [f-XXXXXXXX] where X is hex (the first 8 chars of a UUID).
FINDING_ID_PATTERN = re.compile(r"\[f-([0-9a-f]{8})\]")


def resolve_finding_ids(short_ids: list[str]) -> dict[str, dict]:
    """Look up findings in the DB by their first-8-chars-of-UUID prefix.

    Returns a dict mapping short_id -> finding record (or empty dict if missing).
    This is where the structural anti-hallucination guarantee enforces:
    if the agent fabricates an ID that doesn't exist in the DB, we get
    nothing back and don't render anything for it.
    """
    if not short_ids:
        return {}

    resolved: dict[str, dict] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            for short_id in short_ids:
                # Match against the first 8 chars of the UUID.
                cur.execute(
                    """
                    SELECT id, report_name, source_page, severity, quote
                    FROM findings
                    WHERE id::text LIKE %s
                    LIMIT 1
                    """,
                    (f"{short_id}%",),
                )
                row = cur.fetchone()
                if row:
                    resolved[short_id] = {
                        "id": str(row[0]),
                        "report_name": row[1],
                        "source_page": row[2],
                        "severity": row[3],
                        "quote": row[4],
                    }
    return resolved


@app.command()
def ingest(pdf_path: str):
    """Extract findings from a PDF and store them in the database."""
    run_ingest(pdf_path)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the research corpus"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show agent's tool calls during reasoning"
    ),
):
    """Ask a question. The agent will search the corpus and synthesize an answer."""
    print(f"Question: {question}")
    print("-" * 60)

    # Run the agent
    result = run_agent(question, verbose=verbose)

    # Print the answer
    print()
    print("ANSWER")
    print("-" * 60)
    print(result["answer"])

    # Extract any cited finding IDs and resolve them to quotes
    short_ids = list(set(FINDING_ID_PATTERN.findall(result["answer"])))
    if short_ids:
        resolved = resolve_finding_ids(short_ids)
        print()
        print("CITED FINDINGS")
        print("-" * 60)
        for short_id in short_ids:
            finding = resolved.get(short_id)
            if finding:
                print(f"\n[f-{short_id}] {finding['report_name']} (page {finding['source_page']}, severity={finding['severity']})")
                print(f'  "{finding["quote"]}"')
            else:
                # The agent cited an ID that doesn't exist in the DB.
                # Could be a hallucination, or a typo. Either way, we don't fabricate.
                print(f"\n[f-{short_id}] (citation not found in database)")

    # Footer with provenance
    print()
    print("-" * 60)
    print(
        f"Agent: {result['iterations']} iteration(s), "
        f"{len(result['tool_calls'])} tool call(s), "
        f"terminated by {result['terminated_by']}"
    )


if __name__ == "__main__":
    app()