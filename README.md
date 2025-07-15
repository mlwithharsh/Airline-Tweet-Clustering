# ✈️ Airline Tweet Clustering – Unsupervised Machine Learning Project

**Airline Tweet Clustering** is a real-world NLP + Unsupervised Learning project where tweet data is analyzed to find hidden sentiment patterns and customer behavior using clustering techniques like **KMeans**, **Agglomerative Hierarchical**, and **DBSCAN**.

---

## 📦 Project Overview

This project clusters airline-related tweets using unsupervised learning techniques. It includes:
- End-to-end preprocessing of **raw tweet text**
- Feature extraction using **TF-IDF**
- Encoding of categorical columns like `airline`
- Scaling numerical values like `retweet_count`
- Applying and comparing **3 clustering algorithms**
- Visualizing clusters using **PCA**

---

## 🚀 Algorithms Used & Comparison

| Clustering Model        | Silhouette Score | Notes                              |
|-------------------------|------------------|-------------------------------------|
| **KMeans (K=2)**        | `0.415`          | Best cluster separation             |
| **Agglomerative (K=2)** | `0.409`          | Similar results to KMeans           |
| **DBSCAN**              | `-0.432`         | Poor clustering on TF-IDF text data |

- 📉 **Elbow method** suggested `K=4`, but **Silhouette Score** revealed better performance with `K=2`
- ✅ **KMeans** selected as the best clustering method

---

## 🔧 Tech Stack

| Component          | Tools / Libraries                           |
|--------------------|----------------------------------------------|
| Language           | Python                                       |
| Text Processing    | `nltk`, `re`, `TF-IDF`                       |
| Data Handling      | `pandas`, `numpy`                            |
| Clustering         | `KMeans`, `AgglomerativeClustering`, `DBSCAN`|
| Dimensionality Red.| `PCA`                                        |
| Evaluation         | `silhouette_score`, `inertia`, `elbow method`|
| Visualization      | `matplotlib`, `seaborn`                      |

---

## 📊 Results Summary

```python
#%% md
## 📊 Results Summary:

- **Best Silhouette Score:** KMeans (k=2) → `0.415`
- **Agglomerative Clustering Score:** ~`0.409`
- **DBSCAN:** Tuned `eps` and `min_samples`, but silhouette = `-0.432` (noisy, undefined clusters)
- **Elbow Method** suggested `K=4`, but **K=2** gave the most distinct clusters
- ✅ PCA scatter plots clearly show well-separated clusters


🧠 Key Learnings
Vectorizing raw text with TF-IDF

Feature scaling and encoding for clustering

Comparing multiple clustering models

Visualizing high-dimensional data with PCA

Understanding when and why unsupervised scores may conflict


📁 Airline-Tweet-Clustering/
│
├── 📄 Clustergram.ipynb      # Main notebook with code & analysis
├── 📊 PCA cluster plots      # (Optional: Save screenshots here)
├── 📑 README.md              # Project documentation

🔮 Future Improvements
Add BERT embeddings or Word2Vec instead of TF-IDF

Deploy as a simple dashboard app using Streamlit

Apply topic modeling (LDA) to understand clusters deeper

🌐 Dataset
Source: Kaggle – Airline Tweets Sentiment Dataset

## visit https://github.com/mlwithharsh/Airline-Tweet-Clustering
