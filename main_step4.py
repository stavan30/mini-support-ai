from src.semantic_search import generate_embeddings
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Load the clustered dataset saved from Step 3
    clustered_df = pd.read_csv("data/clustered_df.csv")

    # Combine title + body
    clustered_df['combined_text'] = clustered_df['Title'].fillna('') + ' ' + clustered_df['Body'].fillna('')

    # Step 4: Generate embeddings
    print("\nGenerating embeddings using Sentence-BERT...")
    embeddings = generate_embeddings(clustered_df['combined_text'].tolist())

    # Save for FAISS
    np.save("data/embeddings.npy", embeddings)
    clustered_df.to_csv("data/semantic_search_input.csv", index=False)

    print("Embeddings and dataset saved for semantic search.")
