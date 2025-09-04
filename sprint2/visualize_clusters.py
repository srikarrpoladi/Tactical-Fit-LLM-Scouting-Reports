import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import numpy as np

def plot_pca_clusters(df, features, cluster_col="cluster_label"):
    X = df[features].fillna(0).values 
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    df_plot = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_plot[cluster_col] = df[cluster_col].values
    
    explained_var = pca.explained_variance_ratio_.sum()
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue=cluster_col, palette="tab10")
    plt.title(f"PCA Cluster Plot (Explained Variance: {explained_var:.2f})")
    plt.show()

def plot_radar_for_cluster(df, cluster_label, features):
    cluster_df = df[df["cluster_label"] == cluster_label]
    means = cluster_df[features].fillna(0).mean()  
    
    categories = list(means.index)
    values = means.values
    values = list(values) + [values[0]]  
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]
    
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1) 
    plt.title(f"Cluster {cluster_label} Radar Chart")
    plt.show()
