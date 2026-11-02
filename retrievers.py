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



import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    """Multilingual semantic retrieval using E5 embeddings."""

    def __init__(
        self,
        passages,
        model_name="intfloat/multilingual-e5-small",
    ):
        self.passages = passages.reset_index(drop=True).copy()

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        # E5 uses the passage prefix for documents.
        passage_texts = [
            f"passage: {text}"
            for text in self.passages["text"].tolist()
        ]

        # Encode documents once; reuse these vectors for every query.
        self.passage_embeddings = self.model.encode(
            passage_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return passages ranked by cosine similarity."""
        if not query.strip() or top_k <= 0:
            return []

        # E5 distinguishes queries from passages using prefixes.
        query_embedding = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Normalized embeddings allow dot product to equal cosine similarity.
        scores = self.passage_embeddings @ query_embedding

        ranked_indices = np.argsort(-scores, kind="stable")[
            :min(top_k, len(scores))
        ]

        results = []

        for index in ranked_indices:
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




def clean_asr_query(query: str) -> str:
    """Normalize selected ASR errors and Hinglish phrases."""
    text = query.lower().strip()

    replacements = {
        r"\bfor\s+closure\b": "foreclosure",
        r"\bforclosure\b": "foreclosure",
        r"\bforeclouser\b": "foreclosure",
        r"\bforecloser\b": "foreclosure",
        r"\bmandet\b": "mandate",
        r"\bcharj\b": "charge",
        r"\bcharjes\b": "charges",
        r"\bcharg\b": "charge",
        r"\bstatment\b": "statement",
        r"\bstatments\b": "statements",
        r"\bsertificate\b": "certificate",
        r"\bpaise\s+kat\s+gaya\b": "money deducted",
        r"\bpayment\s+reflect\s+nahi\s+hua\b": "payment not reflected",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Normalize the observed part-payment procedural query.
    if (
        "principal me extra payment request kaha dalu" in text
    ):
        text = text.replace(
            "principal me extra payment request kaha dalu",
            "how to submit a part payment request",
        )

    return text




class CleanedEmbeddingRetriever:
    """Apply conservative query cleanup before E5 retrieval."""

    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        cleaned_query = clean_asr_query(query)
        return self.base_retriever.search(cleaned_query, top_k)



class HybridRetriever:
    """Combine BM25 and E5 rankings using Reciprocal Rank Fusion."""

    def __init__(self, bm25_retriever, embedding_retriever, rrf_k=60):
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []

        # Retrieve enough candidates from each system for fusion.
        candidate_k = min(
            len(self.embedding_retriever.passages),
            max(top_k, 10),
        )

        bm25_results = self.bm25_retriever.search(
            query, top_k=candidate_k
        )
        e5_results = self.embedding_retriever.search(
            query, top_k=candidate_k
        )

        # Store each passage once and accumulate its reciprocal ranks.
        fused = {}

        for results in (bm25_results, e5_results):
            for rank, result in enumerate(results, start=1):
                passage_id = result["passage_id"]

                if passage_id not in fused:
                    fused[passage_id] = {
                        "passage_id": passage_id,
                        "topic": result["topic"],
                        "text": result["text"],
                        "score": 0.0,
                    }

                fused[passage_id]["score"] += (
                    1.0 / (self.rrf_k + rank)
                )

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked[:top_k]


