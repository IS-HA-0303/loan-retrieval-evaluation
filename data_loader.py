
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


if __name__ == "__main__":
    knowledge_base = load_knowledge_base()

    print("Knowledge base validation successful!")
    print(f"Total passages: {len(knowledge_base)}")
    print("\nPassages by topic:")
    print(knowledge_base["topic"].value_counts().sort_index())
    print("\nFirst three passages:")
    print(
        knowledge_base[["passage_id", "topic", "text"]]
        .head(3)
        .to_string(index=False)
    )
