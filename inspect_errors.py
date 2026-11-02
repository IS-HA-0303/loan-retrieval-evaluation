
from data_loader import load_knowledge_base, load_queries
from retrievers import EmbeddingRetriever

knowledge_base = load_knowledge_base()
queries = load_queries()

retriever = EmbeddingRetriever(knowledge_base)

for row in queries.itertuples(index=False):
    if row.query_type == "out_of_scope":
        continue

    results = retriever.search(row.query, top_k=3)
    ranked_ids = [r["passage_id"] for r in results]

    if row.relevant_passage_id not in ranked_ids:
        print("\n" + "=" * 70)
        print(f"ID: {row.query_id}")
        print(f"Type: {row.query_type}")
        print(f"Query: {row.query}")
        print(f"Expected: {row.relevant_passage_id}")
        print(f"Retrieved: {ranked_ids}")

        if results:
            print(f"Top result: {results[0]['text']}")
