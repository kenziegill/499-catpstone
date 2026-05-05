"""Figma REST API integration.

Given a Figma file URL, fetches the file's contents and extracts text from
all text-layer nodes. Used as input to search_findings — pasting a design
file URL becomes a natural-language query.

Requires FIGMA_ACCESS_TOKEN in .env (https://www.figma.com/developers/api#access-tokens).
"""
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

FIGMA_TOKEN = os.environ.get("FIGMA_ACCESS_TOKEN")
FIGMA_API_BASE = "https://api.figma.com/v1"

# Match the file ID in any of these URL formats:
#   https://www.figma.com/file/ABC123XYZ/Name
#   https://www.figma.com/design/ABC123XYZ/Name
#   https://www.figma.com/file/ABC123XYZ/Name?node-id=...
FIGMA_URL_PATTERN = re.compile(
    r"figma\.com/(?:file|design|proto)/([A-Za-z0-9]+)"
)

# Junk patterns we want to exclude from Figma text extraction.
# Designers leave a lot of metadata in template files (URLs, lorem ipsum,
# social handles, attribution lines) that pollute the search query.
_URL_PATTERN = re.compile(r"https?://|www\.")
_SOCIAL_PATTERN = re.compile(r"^[/@]|^[a-zA-Z0-9_]+\d{2,}$")  # /handle, @user, name123
_LOREM_PATTERN = re.compile(r"\blorem\s+ipsum\b", re.IGNORECASE)


def _is_meaningful_ui_text(text: str) -> bool:
    """Filter heuristic: keep text that looks like real UI labels and copy.

    Drops URLs, social handles, lorem ipsum, very short strings, and very
    long marketing paragraphs. Keeps button labels, headings, prices,
    short descriptions — the stuff a designer would point at.
    """
    if not text or len(text) < 2:
        return False
    if len(text) > 200:
        return False  # paragraph-length copy is rarely the design's "intent"
    if _URL_PATTERN.search(text):
        return False
    if _SOCIAL_PATTERN.match(text):
        return False
    if _LOREM_PATTERN.search(text):
        return False
    # Drop strings that are mostly digits or symbols (page numbers, IDs)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < 2:
        return False
    return True


def parse_figma_url(url: str) -> Optional[str]:
    """Extract the Figma file ID from a share URL.

    Returns the file ID (e.g. 'ABC123XYZ') or None if the URL doesn't match.
    """
    match = FIGMA_URL_PATTERN.search(url)
    return match.group(1) if match else None


def fetch_figma_text(url: str) -> dict:
    """Fetch a Figma file and extract its text content.

    Returns a dict with:
      - file_id: extracted file ID
      - file_name: the Figma file's name (from API response)
      - texts: list of text strings found in the file
      - error: None if successful, else a string explaining what went wrong
    """
    if not FIGMA_TOKEN:
        return {
            "file_id": None,
            "file_name": None,
            "texts": [],
            "error": "FIGMA_ACCESS_TOKEN not set in environment",
        }

    file_id = parse_figma_url(url)
    if not file_id:
        return {
            "file_id": None,
            "file_name": None,
            "texts": [],
            "error": f"Could not extract file ID from URL: {url}",
        }

    # Fetch the file from Figma's REST API.
    # X-Figma-Token is the auth header (not Bearer).
    try:
        response = requests.get(
            f"{FIGMA_API_BASE}/files/{file_id}",
            headers={"X-Figma-Token": FIGMA_TOKEN},
            timeout=15,
        )
    except requests.RequestException as e:
        return {
            "file_id": file_id,
            "file_name": None,
            "texts": [],
            "error": f"Network error: {e}",
        }

    if response.status_code == 403:
        return {
            "file_id": file_id,
            "file_name": None,
            "texts": [],
            "error": "Access denied. Verify the token has access to this file.",
        }
    if response.status_code == 404:
        return {
            "file_id": file_id,
            "file_name": None,
            "texts": [],
            "error": "File not found. Check the URL.",
        }
    if not response.ok:
        return {
            "file_id": file_id,
            "file_name": None,
            "texts": [],
            "error": f"Figma API returned {response.status_code}: {response.text[:200]}",
        }

    data = response.json()
    file_name = data.get("name", "Untitled")

    # Walk the document tree and collect text from TEXT nodes.
    # Figma files are deeply nested: document -> pages -> frames -> ... -> text
    texts: list[str] = []

    def walk(node):
        node_type = node.get("type")
        if node_type == "TEXT":
            characters = node.get("characters", "").strip()
            if _is_meaningful_ui_text(characters):
                texts.append(characters)
        for child in node.get("children", []):
            walk(child)

    walk(data.get("document", {}))

    return {
        "file_id": file_id,
        "file_name": file_name,
        "texts": texts,
        "error": None,
    }


def analyze_figma_url(url: str, k: int = 5) -> dict:
    """Fetch a Figma file's text and use it to query the research corpus.

    This is the user-facing function the UI calls. It chains:
      1. fetch_figma_text(url) -> list of UI text from the design
      2. search_findings(joined_text, k) -> findings most relevant to that design

    Returns a dict with:
      - file_name, file_id: from Figma
      - design_text: top text strings extracted (capped for display)
      - findings: list of findings from search_findings (same shape)
      - error: string or None
    """
    fetch_result = fetch_figma_text(url)
    if fetch_result["error"]:
        return {
            "file_name": None,
            "file_id": fetch_result["file_id"],
            "design_text": [],
            "findings": [],
            "error": fetch_result["error"],
        }

    texts = fetch_result["texts"]
    if not texts:
        return {
            "file_name": fetch_result["file_name"],
            "file_id": fetch_result["file_id"],
            "design_text": [],
            "findings": [],
            "error": "No usable text found in this Figma file. Try a file with text labels.",
        }

    # Cap how much we send to the embedder — top 50 unique strings is plenty
    # of signal without burning embedding tokens or noising up the query.
    seen = set()
    unique_texts = []
    for t in texts:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique_texts.append(t)
        if len(unique_texts) >= 50:
            break

    # Join text strings into one query. The embedder will produce a single
    # vector from the joined text — this is the design's "semantic gist."
    query = " ".join(unique_texts)

    # Import here to avoid circular imports at module load
    from tools import search_findings
    findings = search_findings(query=query, k=k)

    return {
        "file_name": fetch_result["file_name"],
        "file_id": fetch_result["file_id"],
        "design_text": unique_texts,
        "findings": findings,
        "error": None,
    }

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python figma.py <figma-url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Analyzing: {url}\n")
    result = analyze_figma_url(url, k=5)

    if result["error"]:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"File: {result['file_name']}")
    print(f"Extracted {len(result['design_text'])} unique UI text strings.")
    print(f"Sample of design text used as query:")
    for t in result["design_text"][:10]:
        print(f"  - {t}")
    print()
    print(f"Top {len(result['findings'])} relevant research findings:")
    for f in result["findings"]:
        print(f"\n  [{f['distance']:.3f}] {f['report_name']} p.{f['source_page']} (severity={f['severity']})")
        print(f"    {f['text'][:120]}...")