
import time
import numpy as np
import pandas as pd

from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever, print_metrics_table
from retrievers import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
)


RANDOM_SEED = 42
TRAIN_FRACTION = 0.5
LATENCY_BUDGET_MS = 150


def split_queries(queries):
    """Create a reproducible, stratified 50/50 query split."""
    selection_parts = []
    test_parts = []

    for _, group in queries.groupby("query_type", sort=True):
        shuffled = group.sample(
            frac=1,
            random_state=RANDOM_SEED,
        )

        midpoint = int(len(shuffled) * TRAIN_FRACTION)
        selection_parts.append(shuffled.iloc[:midpoint])
        test_parts.append(shuffled.iloc[midpoint:])

    selection = (
        pd.concat(selection_parts)
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    test = (
        pd.concat(test_parts)
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    return selection, test


def get_top_score(retriever, query, top_k):
    """Return the highest retrieval score, or zero if no result exists."""
    results = retriever.search(query, top_k=top_k)

    if not results:
        return 0.0

    return max(float(result["score"]) for result in results)


def select_threshold(selection_queries, retriever, top_k):
    """
    Select a threshold using only the selection split.

    Refuse a query when its highest retrieval score is below
    the threshold. Optimize balanced accuracy so that answerable
    and out-of-scope queries receive equal weight.
    """
    examples = []

    for row in selection_queries.itertuples(index=False):
        score = get_top_score(retriever, row.query, top_k)

        examples.append({
            "score": score,
            "is_out_of_scope": row.query_type == "out_of_scope",
        })

    scores = sorted({item["score"] for item in examples})

    candidates = [scores[0] - 1e-9]

    candidates.extend(
        (left + right) / 2
        for left, right in zip(scores, scores[1:])
    )

    candidates.append(scores[-1] + 1e-9)

    best_threshold = None
    best_balanced_accuracy = -1.0

    for threshold in candidates:
        correct_refusals = 0
        total_out_of_scope = 0
        correct_answers = 0
        total_answerable = 0

        for item in examples:
            refused = item["score"] < threshold

            if item["is_out_of_scope"]:
                total_out_of_scope += 1
                if refused:
                    correct_refusals += 1
            else:
                total_answerable += 1
                if not refused:
                    correct_answers += 1

        refusal_rate = correct_refusals / total_out_of_scope
        answer_rate = correct_answers / total_answerable

        balanced_accuracy = (refusal_rate + answer_rate) / 2

        # Prefer the lower threshold on ties to reduce
        # unnecessary refusals of answerable queries.
        if (
            balanced_accuracy > best_balanced_accuracy
            or (
                balanced_accuracy == best_balanced_accuracy
                and (
                    best_threshold is None
                    or threshold < best_threshold
                )
            )
        ):
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold

    return best_threshold, best_balanced_accuracy


def evaluate_threshold(test_queries, retriever, top_k, threshold):
    """Evaluate threshold performance on held-out queries."""
    correct_refusals = 0
    total_out_of_scope = 0
    wrong_refusals = 0
    total_answerable = 0

    for row in test_queries.itertuples(index=False):
        score = get_top_score(retriever, row.query, top_k)
        refused = score < threshold

        if row.query_type == "out_of_scope":
            total_out_of_scope += 1

            if refused:
                correct_refusals += 1
        else:
            total_answerable += 1

            if refused:
                wrong_refusals += 1

    correct_refusal_pct = (
        100 * correct_refusals / total_out_of_scope
        if total_out_of_scope else 0.0
    )

    wrong_refusal_pct = (
        100 * wrong_refusals / total_answerable
        if total_answerable else 0.0
    )

    print("\nHeld-out no-answer threshold evaluation")
    print("-" * 70)
    print(f"Threshold: {threshold:.8f}")
    print(
        f"Out-of-scope queries correctly refused: "
        f"{correct_refusals}/{total_out_of_scope} "
        f"({correct_refusal_pct:.1f}%)"
    )
    print(
        f"Answerable queries wrongly refused: "
        f"{wrong_refusals}/{total_answerable} "
        f"({wrong_refusal_pct:.1f}%)"
    )


def measure_latency(retriever, queries, top_k):
    """
    Measure warm CPU search latency in milliseconds.

    Model initialization is excluded. Search includes query
    embedding generation where applicable.
    """
    query_texts = queries["query"].tolist()

    if not query_texts:
        return 0.0

    retriever.search(query_texts[0], top_k=top_k)

    latencies_ms = []

    for query in query_texts:
        start = time.perf_counter()
        retriever.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    return float(np.percentile(latencies_ms, 95))


def print_failure_examples(retriever, queries, limit=8):
    """
    Show answerable queries whose correct passage is missing
    from the top three retrieved results.
    """
    failures = []

    for row in queries.itertuples(index=False):
        if row.query_type == "out_of_scope":
            continue

        results = retriever.search(row.query, top_k=3)

        retrieved_ids = [
            result["passage_id"]
            for result in results
        ]

        if row.relevant_passage_id not in retrieved_ids:
            failures.append({
                "query_id": row.query_id,
                "query_type": row.query_type,
                "query": row.query,
                "expected": row.relevant_passage_id,
                "retrieved": retrieved_ids,
            })

    print("\n" + "=" * 70)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 70)

    print(
        "Answerable queries missing their correct passage "
        f"from the top 3: {len(failures)}"
    )

    if not failures:
        print("No top-3 retrieval failures found.")
        return

    for failure in failures[:limit]:
        print(f"\nQuery ID: {failure['query_id']}")
        print(f"Type: {failure['query_type']}")
        print(f"Query: {failure['query']}")
        print(f"Expected passage: {failure['expected']}")
        print(
            f"Retrieved passages: "
            f"{failure['retrieved']}"
        )


def main():
    print("=" * 70)
    print("LOAN RETRIEVAL EVALUATION")
    print("=" * 70)

    # 1. Load and validate data.
    knowledge_base = load_knowledge_base()
    queries = load_queries()

    print("\nDataset validation successful!")
    print(f"Knowledge base passages: {len(knowledge_base)}")
    print(f"Total queries: {len(queries)}")

    print("\nQuery distribution:")
    print(
        queries["query_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # 2. Initialize retrieval systems.
    print("\n" + "=" * 70)
    print("INITIALIZING RETRIEVERS")
    print("=" * 70)

    print("\nLoading BM25...")
    bm25 = BM25Retriever(knowledge_base)
    print("BM25 ready.")

    print("\nLoading multilingual E5...")
    e5 = EmbeddingRetriever(knowledge_base)
    print("E5 ready.")

    hybrid = HybridRetriever(bm25, e5)
    print("Hybrid BM25 + E5 ready.")

    systems = {
        "BM25": bm25,
        "Multilingual E5": e5,
        "Hybrid BM25 + E5": hybrid,
    }

    # 3. Evaluate standard retrieval metrics.
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS")
    print("=" * 70)

    for system_name, retriever in systems.items():
        print(f"\nEvaluating {system_name}...")

        summary = evaluate_retriever(
            retriever,
            queries,
            knowledge_base,
        )

        print_metrics_table(summary, system_name)

    # 4. Measure hybrid retrieval CPU latency.
    print("\n" + "=" * 70)
    print("CPU LATENCY")
    print("=" * 70)

    p95_ms = measure_latency(
        hybrid,
        queries,
        top_k=len(knowledge_base),
    )

    print(f"Queries measured: {len(queries)}")
    print(f"Warm search p95 latency: {p95_ms:.2f} ms")
    print(f"Latency budget: {LATENCY_BUDGET_MS} ms")

    if p95_ms <= LATENCY_BUDGET_MS:
        print("Result: Within the 150 ms budget.")
    else:
        print("Result: Exceeds the 150 ms budget.")

    # 5. Select the no-answer threshold using half the queries.
    print("\n" + "=" * 70)
    print("NO-ANSWER THRESHOLD")
    print("=" * 70)

    selection_queries, test_queries = split_queries(queries)

    print(
        f"Threshold-selection queries: "
        f"{len(selection_queries)}"
    )
    print(f"Held-out test queries: {len(test_queries)}")

    top_k = len(knowledge_base)

    threshold, selection_score = select_threshold(
        selection_queries,
        hybrid,
        top_k,
    )

    print(
        f"Selection balanced accuracy: "
        f"{selection_score:.3f}"
    )

    # 6. Evaluate threshold on held-out queries only.
    evaluate_threshold(
        test_queries,
        hybrid,
        top_k,
        threshold,
    )

    # 7. Inspect retrieval failures for the README.
    print_failure_examples(
        hybrid,
        queries,
        limit=8,
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()


