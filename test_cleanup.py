
import time

import numpy as np

from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever, print_metrics_table
from retrievers import (
    EmbeddingRetriever,
    CleanedEmbeddingRetriever,
    clean_asr_query,
)


knowledge_base = load_knowledge_base()
queries = load_queries()

print("Loading E5 model...")
base_retriever = EmbeddingRetriever(knowledge_base)
improved_retriever = CleanedEmbeddingRetriever(base_retriever)

print("\nChecking cleanup examples:")

examples = [
    "for closure charges kitne hai",
    "foreclouser statement kaha se milega",
    "loan statment download kaha se karna hai",
    "What are the foreclosure charges?",
]

for query in examples:
    print(f"Original: {query}")
    print(f"Cleaned:  {clean_asr_query(query)}\n")


baseline = evaluate_retriever(
    base_retriever, queries, knowledge_base
)

improved = evaluate_retriever(
    improved_retriever, queries, knowledge_base
)

print_metrics_table(baseline, "E5 baseline")
print_metrics_table(improved, "E5 + query cleanup")


print("\nRecall@1 improvement:")
for query_type in sorted(baseline):
    old = baseline[query_type]["recall_at_1"]
    new = improved[query_type]["recall_at_1"]

    print(
        f"{query_type}: "
        f"{old:.3f} -> {new:.3f} "
        f"({(new - old) * 100:+.1f} percentage points)"
    )


# Warm up query encoding before measuring latency.
for query in queries["query"].head(5):
    improved_retriever.search(query, top_k=3)

latencies_ms = []

for query in queries["query"]:
    start = time.perf_counter()
    improved_retriever.search(query, top_k=3)
    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies_ms.append(elapsed_ms)

p95_ms = float(np.percentile(latencies_ms, 95))

print("\nCPU latency:")
print(f"Queries measured: {len(latencies_ms)}")
print(f"p50: {np.percentile(latencies_ms, 50):.2f} ms")
print(f"p95: {p95_ms:.2f} ms")
print(f"Maximum: {max(latencies_ms):.2f} ms")
print(f"Assignment latency budget: 150 ms")
print(f"Within budget: {p95_ms <= 150}")
