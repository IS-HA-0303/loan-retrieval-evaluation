from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever
from retrievers import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
)


def main():
    print("=" * 90)
    print("RETRIEVAL SYSTEM COMPARISON")
    print("=" * 90)

    # Step 1: Load the validated datasets.
    knowledge_base = load_knowledge_base()
    queries = load_queries()

    print(f"\nKnowledge base passages: {len(knowledge_base)}")
    print(f"Evaluation queries: {len(queries)}")

    # Step 2: Initialize all three retrieval systems.
    print("\nInitializing retrieval systems...")

    bm25 = BM25Retriever(knowledge_base)
    e5 = EmbeddingRetriever(knowledge_base)
    hybrid = HybridRetriever(bm25, e5)

    systems = {
        "BM25": bm25,
        "Multilingual E5": e5,
        "Hybrid BM25 + E5": hybrid,
    }

    # Step 3: Evaluate each system using the same queries.
    all_results = {}

    for name, retriever in systems.items():
        print(f"Evaluating {name}...")

        all_results[name] = evaluate_retriever(
            retriever,
            queries,
            knowledge_base,
        )

    # Step 4: Print a side-by-side comparison for each query type.
    query_types = sorted(
        next(iter(all_results.values())).keys()
    )

    metrics = ["recall_at_1", "recall_at_3", "mrr"]

    for query_type in query_types:
        print("\n" + "=" * 90)
        print(f"QUERY TYPE: {query_type}")
        print("=" * 90)

        print(
            f"{'System':<22}"
            f"{'Queries':>10}"
            f"{'Recall@1':>14}"
            f"{'Recall@3':>14}"
            f"{'MRR':>12}"
        )
        print("-" * 72)

        for name, results in all_results.items():
            values = results.get(query_type)

            if values is None:
                continue

            print(
                f"{name:<22}"
                f"{values['queries']:>10}"
                f"{values['recall_at_1']:>14.3f}"
                f"{values['recall_at_3']:>14.3f}"
                f"{values['mrr']:>12.3f}"
            )

    # Step 5: Calculate macro-averaged metrics across answerable types.
    print("\n" + "=" * 90)
    print("OVERALL COMPARISON (MACRO-AVERAGE)")
    print("=" * 90)

    print(
        f"{'System':<22}"
        f"{'Recall@1':>14}"
        f"{'Recall@3':>14}"
        f"{'MRR':>12}"
    )
    print("-" * 62)

    for name, results in all_results.items():
        available_types = [
            results[qtype]
            for qtype in results
            if qtype != "out_of_scope"
        ]

        if not available_types:
            continue

        averages = {
            metric: sum(row[metric] for row in available_types)
            / len(available_types)
            for metric in metrics
        }

        print(
            f"{name:<22}"
            f"{averages['recall_at_1']:>14.3f}"
            f"{averages['recall_at_3']:>14.3f}"
            f"{averages['mrr']:>12.3f}"
        )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
