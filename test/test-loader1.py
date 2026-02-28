from data_loader import load_documents

docs = load_documents(r"C:\Users\DELL\Downloads\enzymes-20260227T203523Z-1-001\enzymes")

for doc in docs:
    print("=" * 60)
    print(f"ID    : {doc['id']}")
    print(f"Titre : {doc['title']}")
    print(f"Mots  : {len(doc['content'].split())}")
    print(f"Texte :\n{doc['content'][:500]}")  # 500 premiers caractères
    print()