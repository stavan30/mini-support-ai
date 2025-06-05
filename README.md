# 🔍 Mini Support AI – Semantic Search for Technical Issues

This project builds a **semantic search engine** over support issues/questions, similar to Jira tickets or Stack Overflow posts. It walks through data preprocessing, clustering, semantic embeddings, and a Streamlit-based UI for search.

---

## ✅ Goal

Help users find relevant past issues **based on meaning**, not just keywords — useful for support teams managing tools like Docker, Git, Artifactory, Bitbucket, etc.

---

## 🗂️ Dataset Used

We used a high-quality subset of 60K Stack Overflow questions:
- Source: [Kaggle - 60k Stack Overflow Questions](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)
- Filtered only `HQ` (High-Quality) entries for cleaner training

---

## 🧱 Project Structure

mini-support-ai/
├── main.py # Step 1–3: Load, clean, cluster
├── main_step4.py # Step 4: Embedding via Sentence-BERT
├── main_step5.py # Step 5: Search via FAISS
├── app.py # Step 6: Streamlit UI
├── data/ # Datasets and embeddings
├── src/ # Modular Python files
└── requirements.txt

---

## 🔢 Step-by-Step Pipeline

### ✅ Step 1: Load and Filter High-Quality Questions

- File: `main.py` → `load_high_quality_questions()`
- Loads `train.csv` and filters only rows with label `"HQ"`
- Result: ~15,000 well-written technical questions

**Why?**
To simulate well-documented Jira tickets — clean and useful data only.

---

### ✅ Step 2: Clean the Text

- File: `main.py` → `clean_dataset()`
- Uses BeautifulSoup and regex to remove:
  - HTML tags from the `Body`
  - Converts `Tags` like `<python><docker>` → `["python", "docker"]`

**Why?**
Clean text improves both clustering and semantic embeddings later.

---

### ✅ Step 3: Categorize Using KMeans Clustering (`k=7`)

- File: `main.py` → `cluster_questions()`
- Combined `Title` + `Body` → TF-IDF → KMeans
- `k=7` gave optimal separation after testing `k=5` and `k=6`

**Why?**
Grouping questions into meaningful topics helps users browse and filter.

**Cluster Breakdown (Example):**

| Cluster | Topic                        | Top Terms                             |
|---------|------------------------------|----------------------------------------|
| 0       | Swift / Language Concepts    | string, class, function, return        |
| 1       | React / Frontend             | react, component, js, webpack          |
| 3       | Java + Data Tools            | java, pandas, dataframe, spark         |
| 5       | Android Development          | android, gradle, build, studio         |
| 6       | Docker / Containerization    | docker, compose, container, run        |

---

### ✅ Step 4: Generate Embeddings using Sentence-BERT

- File: `main_step4.py`
- Used `all-MiniLM-L6-v2` model from `sentence-transformers`
- Output: 384-dimension embeddings stored in `embeddings.npy`

**Why?**
Embeddings capture the *meaning* of each question → enables semantic search.

---

### ✅ Step 5: Semantic Search with FAISS

- File: `main_step5.py`
- Loads embeddings into FAISS index
- Accepts user query → encodes it → returns closest matching issues

**Why?**
Allows matching questions even when words differ, but meanings are similar.

Example:
Query: "docker build fails"
Matches:

1-add failed: no such file/directory

2-docker issue: /bin/sh: pip: not found

3-error pulling image: unexpected EOF

---

## ▶️ Run the App

```bash
streamlit run app.py

# Install with:

pip install -r requirements.txt
