from src.faiss_search import SemanticSearchEngine

if __name__ == "__main__":
    engine = SemanticSearchEngine(
        csv_path="data/semantic_search_input.csv",
        embeddings_path="data/embeddings.npy"
    )

    print("\nSemantic Search Ready. Type a question (or 'exit' to quit):")
    while True:
        query = input("\n Your query: ")
        if query.lower() == "exit":
            break

        results = engine.search(query, top_k=5)
        print("\nTop Matches:")
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] {res['title']} (Cluster: {res['cluster']})")
