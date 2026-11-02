from collections import defaultdict


def recall_at_k(ranked_passage_ids, relevant_passage_id, k):
    """Return 1 if the relevant passage occurs in the top k."""
    return int(relevant_passage_id in ranked_passage_ids[:k])


def reciprocal_rank(ranked_passage_ids, relevant_passage_id):
    """Return the reciprocal rank of the relevant passage."""
    for rank, passage_id in enumerate(ranked_passage_ids, start=1):
        if passage_id == relevant_passage_id:
            return 1.0 / rank

    return 0.0


def evaluate_retriever(retriever, queries, knowledge_base):
    """
    Evaluate a retriever on all answerable queries.

    Returns Recall@1, Recall@3, and MRR separately
    for each answerable query type.
    """
    results_by_type = defaultdict(list)

    # Search the entire knowledge base so MRR is not
    # artificially limited to the top three results.
    top_k = len(knowledge_base)

    for row in queries.itertuples(index=False):
        # Out-of-scope questions have no relevant passage.
        if row.query_type == "out_of_scope":
            continue

        results = retriever.search(row.query, top_k=top_k)

        ranked_ids = [
            result["passage_id"]
            for result in results
        ]

        relevant_id = row.relevant_passage_id

        results_by_type[row.query_type].append(
            {
                "recall_at_1": recall_at_k(
                    ranked_ids, relevant_id, 1
                ),
                "recall_at_3": recall_at_k(
                    ranked_ids, relevant_id, 3
                ),
                "mrr": reciprocal_rank(
                    ranked_ids, relevant_id
                ),
            }
        )

    summary = {}

    for query_type, rows in sorted(results_by_type.items()):
        count = len(rows)

        summary[query_type] = {
            metric: sum(row[metric] for row in rows) / count
            for metric in ("recall_at_1", "recall_at_3", "mrr")
        }
        summary[query_type]["queries"] = count

    return summary


def print_metrics_table(summary, system_name):
    """Print evaluation results in a readable table."""
    print(f"\n{system_name} — Retrieval Metrics")
    print("-" * 76)

    print(
        f"{'Query type':<16}"
        f"{'Queries':>9}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'MRR':>12}"
    )

    for query_type, metrics in summary.items():
        print(
            f"{query_type:<16}"
            f"{metrics['queries']:>9}"
            f"{metrics['recall_at_1']:>12.3f}"
            f"{metrics['recall_at_3']:>12.3f}"
            f"{metrics['mrr']:>12.3f}"
        )
