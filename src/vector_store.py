from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_vectorstore(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

def search(query, model, index, chunks, k=5):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    for d, i in zip(distances[0], indices[0]):
        print("DIST:", d)
        print(chunks[i][:300])
        print("-"*60)

    results = [chunks[i] for i in indices[0]]
    return results
