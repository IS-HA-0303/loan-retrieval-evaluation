
from data_loader import load_knowledge_base

knowledge_base = load_knowledge_base()

print("Available columns:", knowledge_base.columns.tolist())
print("Total passages:", len(knowledge_base))

target_ids = ["P07", "P08", "P16", "P17", "P20"]

for passage_id in target_ids:
    matches = knowledge_base[
        knowledge_base["passage_id"].astype(str).str.strip() == passage_id
    ]

    print("\n" + "=" * 70)

    if matches.empty:
        print(f"{passage_id}: NOT FOUND")
        continue

    for _, row in matches.iterrows():
        print(f"Passage ID: {row['passage_id']}")
        print(f"Topic: {row['topic']}")
        print(f"Text: {row['text']}")

