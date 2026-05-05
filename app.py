"""Streamlit UI for Readout — wraps the CLI agent and ingest in a web app.

Run with: streamlit run app.py

This is a thin presentation layer over the same agent.py and ingest.py
the CLI uses. The core logic is unchanged.
"""
import re
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import run_agent
from db import get_conn
from ingest import ingest as run_ingest
from cli import FINDING_ID_PATTERN, resolve_finding_ids

load_dotenv()

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Readout — UX Research Analyst",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Readout")
st.caption("A UX research analyst agent that reasons across your research corpus.")


# -----------------------------------------------------------------------------
# Sidebar: corpus stats + ingest
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Corpus")

    # Show what's currently indexed
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT report_name, COUNT(*) FROM findings GROUP BY report_name ORDER BY report_name"
                )
                report_counts = cur.fetchall()
                cur.execute("SELECT COUNT(*) FROM findings")
                total = cur.fetchone()[0]
    except Exception as e:
        st.error(f"Database error: {e}")
        report_counts = []
        total = 0

    st.metric("Total findings", total)

    if report_counts:
        st.write("**Indexed reports:**")
        for name, count in report_counts:
            st.write(f"• {name} ({count} findings)")
    else:
        st.write("*No reports indexed yet*")

    st.divider()

    st.header("Ingest a new report")
    uploaded = st.file_uploader("Upload a PDF research report", type=["pdf"])
    if uploaded is not None:
        if st.button("Ingest"):
            # Streamlit gives us an UploadedFile in memory; write it to a temp
            # path so PyMuPDF can open it like a real file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    run_ingest(tmp_path)
                    st.success(f"Ingested {uploaded.name}. Refresh to see updated counts.")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
                finally:
                    os.unlink(tmp_path)


# -----------------------------------------------------------------------------
# Main area: ask a question
# -----------------------------------------------------------------------------

st.header("Ask a question")

# A few preset questions to make the demo flow easier
preset = st.selectbox(
    "Pick an example question (or type your own below)",
    [
        "",
        "What do we know about mobile checkout?",
        "Do our studies agree on where to place the primary CTA?",
        "What did participants say about error messages?",
        "What's the most severe issue users had with checkout?",
        "Has anyone tested the onboarding flow?",
    ],
)

question = st.text_input(
    "Your question",
    value=preset,
    placeholder="e.g. What do we know about mobile checkout?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("The agent is thinking..."):
        result = run_agent(question, verbose=False)

    # Show the synthesized answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Show metadata about the agent's run (collapsed by default)
    with st.expander("Agent reasoning details"):
        st.write(f"**Iterations:** {result['iterations']}")
        st.write(f"**Tool calls:** {len(result['tool_calls'])}")
        st.write(f"**Terminated by:** `{result['terminated_by']}`")
        if result["tool_calls"]:
            st.write("**Tool sequence:**")
            for i, (name, inp) in enumerate(result["tool_calls"], start=1):
                st.code(f"{i}. {name}({inp})", language="python")

    # Resolve cited finding IDs to full quotes from the database
    short_ids = list(set(FINDING_ID_PATTERN.findall(result["answer"])))
    if short_ids:
        resolved = resolve_finding_ids(short_ids)
        st.subheader("Cited findings")
        st.caption(
            "Quotes are fetched verbatim from the database by finding ID. "
            "The agent never outputs quote text directly — this is the structural "
            "guarantee against quote hallucination."
        )

        for short_id in short_ids:
            finding = resolved.get(short_id)
            if finding:
                with st.container(border=True):
                    st.write(
                        f"**[f-{short_id}]** "
                        f"`{finding['report_name']}` · "
                        f"page {finding['source_page']} · "
                        f"severity: {finding['severity']}"
                    )
                    st.write(f"> {finding['quote']}")
            else:
                st.warning(f"[f-{short_id}] citation not found in database")
    else:
        st.info("This response did not cite any specific findings.")


# -----------------------------------------------------------------------------
# Figma analysis: paste a design URL, get research findings that apply
# -----------------------------------------------------------------------------

st.divider()
st.header("Or — analyze a Figma design")
st.caption(
    "Paste a Figma file URL and Readout will fetch the design's text content "
    "and surface the most relevant findings from the research corpus. Useful "
    "when you're working on a specific screen and want to know what research applies."
)

figma_url = st.text_input(
    "Figma file URL",
    placeholder="https://www.figma.com/design/...",
    key="figma_url_input",
)

if st.button("Analyze design") and figma_url.strip():
    with st.spinner("Fetching Figma file and searching research corpus..."):
        from figma import analyze_figma_url
        figma_result = analyze_figma_url(figma_url, k=5)

    if figma_result["error"]:
        st.error(f"Could not analyze: {figma_result['error']}")
    else:
        st.success(f"Analyzed: **{figma_result['file_name']}**")

        # Show what text was extracted from the design (collapsed)
        with st.expander(
            f"Design text extracted ({len(figma_result['design_text'])} unique strings)"
        ):
            st.write(", ".join(figma_result["design_text"]))

        # Show the top relevant findings
        st.subheader("Most relevant research findings")
        st.caption(
            "These findings rank highest in semantic similarity to the text "
            "content of your design file. Quotes are fetched verbatim from the "
            "database — same structural anti-hallucination guarantee as the "
            "question-answering flow."
        )

        if not figma_result["findings"]:
            st.info("No findings found.")
        else:
            for f in figma_result["findings"]:
                with st.container(border=True):
                    st.write(
                        f"**`{f['report_name']}`** · "
                        f"page {f['source_page']} · "
                        f"severity: {f['severity']} · "
                        f"distance: {f['distance']:.3f}"
                    )
                    st.write(f["text"])
                    st.write(f"> {f['quote']}")