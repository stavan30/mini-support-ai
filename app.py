import streamlit as st
from src.faiss_search import SemanticSearchEngine

# Initialize engine (only once)
@st.cache_resource
def load_engine():
    return SemanticSearchEngine(
        csv_path="data/semantic_search_input.csv",
        embeddings_path="data/embeddings.npy"
    )

engine = load_engine()

st.title("Semantic Support Search")
st.subheader("Find similar support issues using natural language")

query = st.text_input("Enter your question")

if query:
    top_k = st.slider("Number of results", 1, 10, 5)
    results = engine.search(query, top_k=top_k)

    st.markdown("### Results")
    for i, res in enumerate(results, 1):
        st.markdown(f"**{i}. {res['title']}**")
        st.markdown(f"`Cluster: {res['cluster']}`")
        st.markdown(f"<small>{res['body'][:300]}...</small>", unsafe_allow_html=True)
        st.markdown("---")
