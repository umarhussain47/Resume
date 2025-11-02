import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Configuration
# -------------------------------

# Path to CSV dataset (replace with your local or S3 path)
DATA_PATH = "data/raw/resumes.csv"  # CSV with 'category' & 'resume' columns

# MLflow tracking URI (EC2-hosted)
MLFLOW_URI = "http://<EC2_PUBLIC_IP>:5000"  # Replace with your EC2 public IP

# TF-IDF parameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# -------------------------------
# 2. Helper Functions
# -------------------------------

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercasing, removing special chars, extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# 3. Load & Preprocess Data
# -------------------------------

df = pd.read_csv(DATA_PATH)

# Check required columns
if not {"category", "resume"}.issubset(df.columns):
    raise ValueError("CSV must have 'category' and 'resume' columns.")

df["resume_clean"] = df["resume"].apply(clean_text)
categories = df["category"].unique().tolist()
resumes = df["resume_clean"].tolist()

# -------------------------------
# 4. Train-Test Split (Optional)
# -------------------------------

X_train, X_test = train_test_split(resumes, test_size=0.2, random_state=42)

# -------------------------------
# 5. MLflow Setup
# -------------------------------

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Resume_Matching")

with mlflow.start_run(run_name="TFIDF_Cosine_Model_v1"):

    # -------------------------------
    # 6. TF-IDF Vectorization
    # -------------------------------
    vectorizer = TfidfVectorizer(stop_words="english",
                                 max_features=MAX_FEATURES,
                                 ngram_range=NGRAM_RANGE)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------------------
    # 7. Similarity Computation (Optional)
    # -------------------------------
    # Compute similarity between first test resume and all train resumes
    similarity_matrix = cosine_similarity(X_test_tfidf, X_train_tfidf)
    avg_similarity = np.mean(similarity_matrix)

    # -------------------------------
    # 8. MLflow Logging
    # -------------------------------
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("ngram_range", NGRAM_RANGE)
    mlflow.log_metric("avg_similarity", avg_similarity)

    # Log TF-IDF vectorizer as model
    mlflow.sklearn.log_model(vectorizer, artifact_path="tfidf_vectorizer")

    # Log similarity matrix as artifact
    np.save("similarity_matrix.npy", similarity_matrix)
    mlflow.log_artifact("similarity_matrix.npy")

    print("✅ TF-IDF vectorizer and similarity matrix logged to MLflow.")
    print("Average similarity:", avg_similarity)

print("✅ Training complete and MLflow run finished.")
