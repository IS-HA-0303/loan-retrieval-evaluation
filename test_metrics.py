
from data_loader import load_knowledge_base, load_queries
from metrics import evaluate_retriever, print_metrics_table
from retrievers import BM25Retriever


knowledge_base = load_knowledge_base()
queries = load_queries()

retriever = BM25Retriever(knowledge_base)

summary = evaluate_retriever(
    retriever,
    queries,
    knowledge_base,
)

print_metrics_table(summary, "BM25")
