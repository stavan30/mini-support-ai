from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(text_list, model_name='all-MiniLM-L6-v2'):
    """
    Converts a list of texts into embeddings using Sentence-BERT.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings
