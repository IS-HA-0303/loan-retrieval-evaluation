import re

from rank_bm25 import BM25Okapi

from data_loader import load_knowledge_base


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Keyword-based retrieval over the loan knowledge base."""

    def __init__(self, passages):
        self.passages = passages.reset_index(drop=True).copy()

        # Tokenize each passage once when the index is created.
        tokenized_passages = [
            tokenize(text)
            for text in self.passages["text"]
        ]

        self.index = BM25Okapi(tokenized_passages)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the highest-scoring passages for a query."""
        if not query.strip():
            return []

        if top_k <= 0:
            return []

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.index.get_scores(tokens)

        # Sort by score from highest to lowest.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            passage = self.passages.iloc[index]

            results.append(
                {
                    "passage_id": passage["passage_id"],
                    "topic": passage["topic"],
                    "text": passage["text"],
                    "score": float(scores[index]),
                }
            )

        return results


if __name__ == "__main__":
    knowledge_base = load_knowledge_base()
    retriever = BM25Retriever(knowledge_base)

    test_queries = [
        "What are the foreclosure charges?",
        "How much is the part payment charge?",
        "When is my EMI due each month?",
        "Where can I download my loan statement?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        for rank, result in enumerate(
            retriever.search(query, top_k=3),
            start=1,
        ):
            print(
                f"{rank}. {result['passage_id']} "
                f"| Score: {result['score']:.4f} "
                f"| Topic: {result['topic']}"
            )
