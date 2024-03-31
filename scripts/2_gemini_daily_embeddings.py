#!/usr/bin/env python
# coding: utf-8

# ## 1. Load required libraries

# In[ ]:


import os
import time
import json
import gzip
import google.generativeai as genai

from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = "gemini-1.0-pro-001"


# ## 2. Define LLM functions

# In[ ]:


genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(GEMINI_EMBEDDING_MODEL, safety_settings={ # #gemini-1.0-pro
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE})


def get_embeddings_from_gemini_by_chunk(abstracts: list[str], task_type: str="clustering", 
                                        chunk_size: int=100, max_retries: int=3) -> list[list[float]]:
    
    embeddings = []

    for i in range(0, len(abstracts), chunk_size):
        chunk = abstracts[i : i + chunk_size]
        retries = 0
        while retries < max_retries:
            try:
                chunk_embeddings = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type=task_type,
                )

                embeddings.extend(chunk_embeddings["embedding"])
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(10)

    return embeddings


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
    abstracts = [d["abstract"] for d in arxiv]
    embeddings = get_embeddings_from_gemini_by_chunk(abstracts)
    for embedding, entry in zip(embeddings, arxiv):
        entry["embedding"] = embedding
        entry["embedding_model"] = GEMINI_EMBEDDING_MODEL
    with gzip.open(f"../data/{filename}", 'w') as file:
        file.write(json.dumps(arxiv).encode('utf-8'))


# ## 3. Retrieve embeddings

# In[ ]:


if __name__ == "__main__":
    daily_processing()

