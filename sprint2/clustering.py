import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def prepare_clustering_features(df):
    df["goalkeeping"] = df["goalkeeping"].fillna(0)
    features = ["attack", "passing", "dribbling", "defense", "physical", "goalkeeping",
                "num_positions", "weak_foot_scaled", "skill_moves_scaled"]
    X = df[features].copy()
    X = X.fillna(X.mean())  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def run_kmeans(X, n_clusters=5, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def evaluate_clusters(X, labels):
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.3f}")
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:", dict(zip(unique, counts)))
    return score

def add_cluster_labels(df, labels):
    df["cluster_label"] = labels
    return df
