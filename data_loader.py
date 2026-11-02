
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.csv"
QUERIES_PATH = DATA_DIR / "queries.csv"


def load_knowledge_base() -> pd.DataFrame:
    """Load and validate the knowledge base CSV."""
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
        )

    df = pd.read_csv(KNOWLEDGE_BASE_PATH)

    required_columns = {"passage_id", "topic", "text"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Knowledge base must contain: {sorted(required_columns)}"
        )

    if len(df) < 20 or len(df) > 25:
        raise ValueError(
            f"Expected 20–25 passages, found {len(df)}"
        )

    if df["passage_id"].isna().any():
        raise ValueError("Some passages have missing IDs.")

    if df["passage_id"].duplicated().any():
        raise ValueError("Duplicate passage IDs found.")

    if df[["topic", "text"]].isna().any().any():
        raise ValueError("Some passages have missing topics or text.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty passage text found.")

    return df


def load_queries() -> pd.DataFrame:
    """Load and validate the 80-query evaluation dataset."""
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Queries file not found: {QUERIES_PATH}"
        )

    df = pd.read_csv(QUERIES_PATH, keep_default_na=False)

    required_columns = {
        "query_id",
        "query",
        "relevant_passage_id",
        "query_type",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Queries CSV must contain: {sorted(required_columns)}"
        )

    if len(df) != 80:
        raise ValueError(f"Expected 80 queries, found {len(df)}")

    if df["query_id"].duplicated().any():
        raise ValueError("Duplicate query IDs found.")

    if df["query"].str.strip().eq("").any():
        raise ValueError("Empty queries found.")

    expected_types = {
        "english_clean",
        "hinglish",
        "asr_noisy",
        "out_of_scope",
    }

    if set(df["query_type"]) != expected_types:
        raise ValueError(
            f"Unexpected query types: {set(df['query_type'])}"
        )

    counts = df["query_type"].value_counts()

    if not (counts == 20).all():
        raise ValueError(
            f"Expected 20 queries per type:\n{counts}"
        )

    knowledge_base = load_knowledge_base()
    valid_ids = set(knowledge_base["passage_id"])

    for row in df.itertuples(index=False):
        passage_id = row.relevant_passage_id

        if row.query_type == "out_of_scope":
            if passage_id != "":
                raise ValueError(
                    f"{row.query_id}: out-of-scope query must "
                    "have an empty relevant_passage_id."
                )
        else:
            if passage_id == "":
                raise ValueError(
                    f"{row.query_id}: answerable query has no label."
                )

            if passage_id not in valid_ids:
                raise ValueError(
                    f"{row.query_id}: unknown passage ID {passage_id}"
                )

    return df




if __name__ == "__main__":
    knowledge_base = load_knowledge_base()
    queries = load_queries()

    print("Dataset validation successful!")
    print(f"Knowledge base passages: {len(knowledge_base)}")
    print(f"Total queries: {len(queries)}")
    print("\nQueries by type:")
    print(queries["query_type"].value_counts().sort_index())
    print("\nFirst five queries:")
    print(
        queries[
            ["query_id", "query", "relevant_passage_id", "query_type"]
        ].head().to_string(index=False)
    )

