#!/usr/bin/env python
# coding: utf-8

# ## 1. Load required libraries

# In[1]:


import os
import json
import gzip
import joblib
import hdbscan
import numpy as np
import matplotlib.pyplot as plt


# ## 2. Define clustering functions

# In[2]:


def get_cluster_labels(embeddings: list[float]) -> list[int]:
    umap_reducer = joblib.load("../models/umap_model.pkl")
    hdbscan_clusterer = joblib.load("../models/hdbscan_model.pkl")
    reduced_embeddings = umap_reducer.transform(embeddings)
    data = hdbscan.approximate_predict(hdbscan_clusterer, reduced_embeddings) 
    labels = data[0].tolist()
    probabilities = data[0].tolist()
    return labels, probabilities


def plot_2d_clusters(embeddings, labels):
    cluster_descriptions = {-1: "Noise",
                            0: "Wireless, Computing",
                            1: "Logic",
                            2: "Privacy, Security",
                            3: "Theoretical Science",
                            4: "Medical Imaging",
                            5: "AI Images", 
                            6: "AI Vision", 
                            7: "Quantum Computing", 
                            8: "RL, Game Theory", 
                            9: "Planning, Control"
                            }
    umap_2d_reducer = joblib.load("../models/umap_2d_model.pkl")
    reduced_embeddings = umap_2d_reducer.transform(embeddings)
    unique_labels = np.unique(labels)
    num_observations = len(labels)
    num_noise = len([num for num in labels if num == -1])
    plt.figure(figsize=(10,8))
    for label in unique_labels:
        cluster_indices = labels == label
        if label == -1:
            plt.scatter(reduced_embeddings[cluster_indices, 0], reduced_embeddings[cluster_indices, 1], label=f"{cluster_descriptions[label]}", color="lightgrey")
        else:
            plt.scatter(reduced_embeddings[cluster_indices, 0], reduced_embeddings[cluster_indices, 1], label=f"{cluster_descriptions[label]}")
    plt.title(label=f"Daily Arxiv Clusters -- obs: {num_observations}; noise: {num_noise}", fontsize=18)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig("../docs/daily_plot.png", bbox_inches='tight')
    plt.close()
    return


def get_most_recent_file(directory: str) -> str | None:
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    file_ctimes = [(f, os.path.getctime(f)) for f in files]
    most_recent_file = sorted(file_ctimes, key=lambda x: x[1], reverse=True)[0][0]
    return most_recent_file


def daily_processing(filename=None):
    if filename is None:
        filename = get_most_recent_file("../data")
    with gzip.open(filename, "r") as file:
        arxiv = json.loads(file.read().decode("utf-8"))
    embeddings = [d["embedding"] for d in arxiv]
    labels, probabilities = get_cluster_labels(embeddings)
    plot_2d_clusters(embeddings, labels)
    return


# ## 3. Daily processing

# In[3]:


if __name__ == "__main__":
    daily_processing()

