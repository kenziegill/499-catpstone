"""Database connection and schema initialization.

Run this once with `python db.py` to create the pgvector extension
and the findings table. Safe to run multiple times — uses IF NOT EXISTS.
"""
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

DATABASE_URL = os.environ["DATABASE_URL"]


def get_conn():
    """Open a Postgres connection. Caller is responsible for closing
    (or use with a context manager, which we do everywhere)."""
    return psycopg.connect(DATABASE_URL)


def init_db():
    """Create the pgvector extension and findings table if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension. Idempotent — safe to run repeatedly.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Findings table. Each row is one extracted finding from one report.
            # Columns:
            #   id           -- UUID, generated client-side, prevents enumeration attacks
            #   report_name  -- which PDF this came from
            #   text         -- model-paraphrased finding (1-2 sentences)
            #   quote        -- verbatim quote from the PDF (verified during ingest)
            #   severity     -- 'low' | 'medium' | 'high', model-assigned
            #   source_page  -- 1-indexed page number
            #   page_text    -- full text of the source page (for get_finding_context tool)
            #   embedding    -- voyage-3 embedding, 1024 dimensions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id           UUID PRIMARY KEY,
                    report_name  TEXT NOT NULL,
                    text         TEXT NOT NULL,
                    quote        TEXT NOT NULL,
                    severity     TEXT,
                    source_page  INT NOT NULL,
                    page_text    TEXT NOT NULL,
                    embedding    VECTOR(1024) NOT NULL
                );
            """)

            # Index for fast cosine-similarity search.
            # ivfflat is good for our scale; lists=10 is fine for small data.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS findings_embedding_idx
                ON findings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10);
            """)
        conn.commit()
    print("Database initialized.")


if __name__ == "__main__":
    init_db()