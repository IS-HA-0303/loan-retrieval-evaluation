
from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever, print_metrics_table
from retrievers import EmbeddingRetriever


knowledge_base = load_knowledge_base()
queries = load_queries()

print("Loading multilingual E5 model...")
retriever = EmbeddingRetriever(knowledge_base)

print("\nTesting individual queries:")

test_queries = [
    "What are the foreclosure charges?",
    "Loan jaldi band karne ka kitna charge lagega?",
    "mera emi har month 5 tarik ko katega",
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

print("\nEvaluating all answerable queries...")

summary = evaluate_retriever(
    retriever,
    queries,
    knowledge_base,
)

print_metrics_table(summary, "Multilingual E5")
