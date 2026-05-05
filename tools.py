"""Tools available to the Research Analyst agent.

Each function in this module corresponds to a tool the agent can call
during its reasoning loop. They're plain Python functions; the agent
schema definitions live in agent.py.
"""
import os
import json
from typing import Literal

import requests
from anthropic import Anthropic
from voyageai import Client as VoyageClient
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from db import get_conn
from pgvector.psycopg import register_vector

load_dotenv()
anthropic_client = Anthropic()
voyage_client = VoyageClient()


# -----------------------------------------------------------------------------
# Tool 1: search_findings
# Datamuse query expansion + voyage-3 embedding + pgvector cosine similarity
# -----------------------------------------------------------------------------

def expand_query_with_datamuse(query: str, max_terms: int = 5) -> str:
    """Use Datamuse 'means like' API to find semantically related terms.

    Datamuse is a free, unauthenticated API that returns words related to
    a search term. We use 'ml' (means-like) to find synonyms and related
    concepts, then append them to the original query before embedding.
    This improves retrieval recall when the user's vocabulary doesn't match
    the corpus vocabulary.

    Falls back gracefully to the original query if the API is unavailable.
    """
    try:
        response = requests.get(
            "https://api.datamuse.com/words",
            params={"ml": query, "max": max_terms},
            timeout=3,
        )
        response.raise_for_status()
        related = [item["word"] for item in response.json()]
        if not related:
            return query
        # Append related terms to the original query so embedding sees both
        expanded = f"{query} {' '.join(related)}"
        return expanded
    except Exception:
        # Datamuse failures should not break search — fall back silently
        return query
def search_findings(query: str, k: int = 5) -> list[dict]:
    """Retrieve the top-k findings from the corpus most relevant to a query."""

    expanded = expand_query_with_datamuse(query)

    embedding_result = voyage_client.embed(
        texts=[expanded],
        model="voyage-3",
        input_type="query",
    )
    query_embedding = embedding_result.embeddings[0]
    query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, report_name, text, quote, severity, source_page, "
                "embedding <=> %s::vector AS distance "
                "FROM findings ORDER BY distance ASC LIMIT %s",
                (query_embedding_str, k),
            )
            rows = cur.fetchall()
    result = [
        {
            "id": str(row[0]),
            "report_name": row[1],
            "text": row[2],
            "quote": row[3],
            "severity": row[4],
            "source_page": row[5],
            "distance": float(row[6]),
        }
        for row in rows
    ]
    return result
# -----------------------------------------------------------------------------
# Tool 2: find_contradictions
# Pairwise reasoning: ask Claude whether any of a set of findings contradict
# -----------------------------------------------------------------------------

class Contradiction(BaseModel):
    finding_a_id: str = Field(description="ID of one finding in the contradicting pair")
    finding_b_id: str = Field(description="ID of the other finding in the pair")
    explanation: str = Field(description="One-sentence explanation of how they contradict")


CONTRADICTION_SYSTEM_PROMPT = """You are a research analyst checking for contradictions between findings.

You will be given a list of findings, each with an ID and text. Identify any pairs of findings that DIRECTLY CONTRADICT each other — meaning they make opposing claims about the same phenomenon, scope, or recommendation.

Do NOT report:
- Findings on different scopes (e.g. mobile vs desktop is NOT a contradiction)
- Findings about different topics
- Findings that simply emphasize different aspects

DO report:
- Findings that make opposing claims within the same scope
- Findings whose recommendations directly conflict

Return JSON matching this schema EXACTLY:
{
  "contradictions": [
    {"finding_a_id": "...", "finding_b_id": "...", "explanation": "..."}
  ]
}

If there are no contradictions, return {"contradictions": []}. Return ONLY the JSON object."""


def find_contradictions(finding_ids: list[str]) -> list[dict]:
    """Check whether any pair of findings (by ID) contradicts.

    Fetches the findings from the DB, sends them to Claude with a
    contradiction-detection prompt, returns validated Contradiction objects.
    Returns empty list on any failure (graceful degradation).
    """
    if len(finding_ids) < 2:
        return []  # need at least 2 findings to compare

    # Fetch findings from DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, report_name, text FROM findings
                WHERE id::text = ANY(%s)
                """,
                (finding_ids,),
            )
            rows = cur.fetchall()

    if len(rows) < 2:
        return []

    # Build a compact prompt input
    findings_text = "\n".join(
        f"- ID: {row[0]} | Report: {row[1]} | Finding: {row[2]}"
        for row in rows
    )

    user_message = f"<findings>\n{findings_text}\n</findings>\n\nCheck for direct contradictions."

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=CONTRADICTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        print(f"[warn] find_contradictions API call failed: {e}")
        return []

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        contradictions = [Contradiction(**c) for c in data.get("contradictions", [])]
        return [c.model_dump() for c in contradictions]
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[warn] find_contradictions output failed validation: {e}")
        return []


# -----------------------------------------------------------------------------
# Tool 3: get_finding_context
# Fetch the full source page text for a finding
# -----------------------------------------------------------------------------

def get_finding_context(finding_id: str) -> dict:
    """Return the full page text where a finding originated.

    Useful when a finding's text is ambiguous and the agent needs more context.
    Returns a dict with id, report_name, source_page, and page_text.
    Returns empty dict if the ID isn't found.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, report_name, source_page, page_text
                FROM findings
                WHERE id::text = %s
                """,
                (finding_id,),
            )
            row = cur.fetchone()

    if not row:
        return {}

    return {
        "id": str(row[0]),
        "report_name": row[1],
        "source_page": row[2],
        "page_text": row[3],
    }


# -----------------------------------------------------------------------------
# Manual test harness — run this file directly to sanity-check the tools
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: search_findings('CTA placement')")
    print("=" * 60)
    results = search_findings("CTA placement", k=3)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['report_name']} p.{r['source_page']}: {r['text'][:80]}...")

    if len(results) >= 2:
        print()
        print("=" * 60)
        print("Test 2: find_contradictions on top 3 search results")
        print("=" * 60)
        ids = [r["id"] for r in results]
        contradictions = find_contradictions(ids)
        if contradictions:
            for c in contradictions:
                print(f"  CONTRADICTION:")
                print(f"    A: {c['finding_a_id']}")
                print(f"    B: {c['finding_b_id']}")
                print(f"    Why: {c['explanation']}")
        else:
            print("  No contradictions found (or API failure).")

        print()
        print("=" * 60)
        print(f"Test 3: get_finding_context({results[0]['id']})")
        print("=" * 60)
        ctx = get_finding_context(results[0]["id"])
        if ctx:
            print(f"  Report: {ctx['report_name']} (page {ctx['source_page']})")
            print(f"  Page text preview: {ctx['page_text'][:200]}...")
        else:
            print("  Not found.")