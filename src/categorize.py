import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def combine_text(row):
    """Combine title and body into one string for clustering."""
    return row['Title'] + ' ' + row['Body']

def cluster_questions(df, num_clusters=5):
    # Combine title + body
    df['combined_text'] = df.apply(combine_text, axis=1)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['combined_text'])

    # Fit KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster_id'] = kmeans.fit_predict(X)

    return df, kmeans, vectorizer
