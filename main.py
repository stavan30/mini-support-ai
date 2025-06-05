from src.load_data import load_high_quality_questions
from src.clean_data import clean_dataset
from src.categorize import cluster_questions
import numpy as np

if __name__ == "__main__":
    data_path = "data/train.csv"

    # Step 1: Load HQ questions
    hq_df = load_high_quality_questions(data_path)

    # Step 2: Clean the data
    clean_df = clean_dataset(hq_df)

    # Step 3: Cluster the questions
    clustered_df, model, vectorizer = cluster_questions(clean_df, num_clusters=7)

    print("\n--- Clustered Sample ---")
    print(clustered_df[['Title', 'cluster_id']].head())

    # Print 3 sample questions per cluster
    print("\n--- Sample Questions per Cluster ---")
    for i in range(clustered_df['cluster_id'].nunique()):
        print(f"\nCluster {i}:")
        samples = clustered_df[clustered_df['cluster_id'] == i]['Title'].head(3)
        for title in samples:
            print(f"  - {title}")

    # Print top 10 keywords per cluster
    print("\n--- Top Terms per Cluster ---")
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for i in range(model.n_clusters):
        print(f"\nCluster {i} top terms:")
        for ind in order_centroids[i, :10]:
            print(f"  {terms[ind]}")
