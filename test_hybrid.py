from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever, print_metrics_table
from retrievers import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
)

knowledge_base = load_knowledge_base()
queries = load_queries()

print("Initializing BM25...")
bm25 = BM25Retriever(knowledge_base)

print("Initializing multilingual E5...")
e5 = EmbeddingRetriever(knowledge_base)

hybrid = HybridRetriever(bm25, e5)

# Inspect the known failure cases.
test_queries = [
    "paise kat gaya but loan me payment reflect nahi hua",
    "Can I pay my EMI in cash at a branch?",
    "principal me extra payment request kaha dalu",
    "What are the foreclosure charges?",
]

for query in test_queries:
    print(f"\nQuery: {query}")

    for rank, result in enumerate(hybrid.search(query, top_k=3), 1):
        print(
            f"{rank}. {result['passage_id']} "
            f"| RRF score: {result['score']:.5f} "
            f"| Topic: {result['topic']}"
        )
        print(f"   {result['text']}")

print("\nEvaluating hybrid retrieval...")
summary = evaluate_retriever(
    hybrid,
    queries,
    knowledge_base,
)

print_metrics_table(summary, "Hybrid BM25 + E5 (RRF)")
