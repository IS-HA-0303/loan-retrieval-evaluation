from retrievers import clean_asr_query

test_queries = [
    "for closure charges kitne hai",
    "loan statment download kaha se karna hai",
    "paise kat gaya but loan me payment reflect nahi hua",
    "principal me extra payment request kaha dalu",
]

for query in test_queries:
    print("Original:", query)
    print("Cleaned: ", clean_asr_query(query))
    print()
