import faiss
import numpy as np
import json
import os

MEMORY_FILE = "memData/memory.json"
FAISS_FILE = "memData/faiss.index"

model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

index = None

def get_index():
    global index
    if index is None:
        import faiss
        index = faiss.IndexFlatL2(384)
    return index

memory_texts = set()
memory_lists = []

def add_memory(text):
    if text in memory_texts:
        return  # avoid duplicates

    embedding = get_model().encode([text])
    get_index().add(np.array(embedding))

    memory_texts.add(text)
    memory_lists.append(text)

    save_memory()

def search_memory(query, k=3):
    if len(memory_texts) == 0:
        return []

    if get_index().ntotal == 0:
        embeddings = get_model().encode(memory_lists)
        get_index().add(np.array(embeddings))

    k = min(k, len(memory_texts))

    embedding = get_model().encode([query])
    D, I = get_index().search(np.array(embedding), k)

    results = []
    for i in I[0]:
        if i < len(memory_lists):
            results.append(memory_lists[i])

    return results

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_lists, f)

def load_memory():
    global memory_lists, memory_texts, index

    if not os.path.exists(MEMORY_FILE):
        return

    with open(MEMORY_FILE, "r") as f:
        memory_lists = json.load(f)

    memory_texts = set(memory_lists)


def save_index():
    faiss.write_index(index, FAISS_FILE)

def load_index():
    global index
    if os.path.exists(FAISS_FILE):
        index = faiss.read_index(FAISS_FILE)