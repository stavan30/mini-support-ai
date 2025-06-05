import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

class SemanticSearchEngine:
    def __init__(self, csv_path, embeddings_path, model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(embeddings_path)
        self.model = SentenceTransformer(model_name)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            row = self.df.iloc[idx]
            results.append({
                "title": row["Title"],
                "cluster": row.get("cluster_id", "N/A"),
                "body": row["Body"]
            })
        return results
