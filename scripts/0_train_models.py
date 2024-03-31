#!/usr/bin/env python
# coding: utf-8

# ## 1. Load required libraries

# In[1]:


import os
import joblib
import umap
import hdbscan
import numpy as np
import pandas as pd


# ## 2. Define training functions

# In[2]:


def get_file_list(directory: str) -> list[str] | None:
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    return files


def get_training_data() -> list[float]:
    df = pd.DataFrame()
    files = get_file_list("../data")
    for file in files:
        df = pd.concat([df, pd.read_json(file, orient="records")])
    return df["embedding"].to_list()


def umap_reducer(n_neighbors=30, min_dist=0.0, n_components=20, metric='cosine') -> umap.UMAP:
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
        )


def umap_reducer_2d(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine') -> umap.UMAP:
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
        )


def hdbscan_clusterer(min_cluster_size=250, min_samples=5, cluster_selection_epsilon=0.0,
                      metric="euclidean", cluster_selection_method="eom", prediction_data=False) -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                           min_samples=min_samples, 
                           cluster_selection_epsilon=cluster_selection_epsilon,
                           metric=metric, 
                           cluster_selection_method=cluster_selection_method, 
                           prediction_data=prediction_data
                           )


def train_models():
    embeddings = get_training_data()
    umap_model = umap_reducer()
    umap_2d_model = umap_reducer_2d()
    hdbscan_model = hdbscan_clusterer()
    reduced_embeddings = umap_model.fit_transform(embeddings)
    labels = hdbscan_model.fit_predict(reduced_embeddings)
    umap_2d_model.fit(embeddings)
    joblib.dump(umap_model, "../models/umap_model.pkl")
    joblib.dump(umap_2d_model, "../models/umap_2d_model.pkl")
    joblib.dump(hdbscan_model, "../models/hdbscan_model.pkl")
    return labels


# ## 3. Train models

# In[3]:


if __name__ == "__main__":
    labels = train_models()
    print("Number of observations:", len(labels))
    print("Number of noise entries:", len([num for num in labels if num == -1]))
    print("Number of labels:", max(labels) + 2)

