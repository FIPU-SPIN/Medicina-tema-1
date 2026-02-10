from pdf_load import load_pdfs
from chunkovi import chunk_text
from vector_store import build_vectorstore, search
from hf_api_llm import generate_definition
from sentence_transformers import SentenceTransformer
import re

def rag_generate_definition(query, pdf_folder="data/raw", top_k=10):
    """
    Input:
        query: str - the term to search for
        pdf_folder: str - folder containing PDF files
        top_k: int - how many top chunks to retrieve
    Output:
        definition: str - generated definition from LLM
    """
    docs = load_pdfs(pdf_folder)
    print(f"Total PDF pages loaded: {len(docs)}")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size=400, overlap=100)
        all_chunks.extend(chunks)
    print(f"Total number of chunks: {len(all_chunks)}")

    clean_chunks = []
    for c in all_chunks:
        if c.count("|") > 5:       
         continue
        if c.count("C1") > 2:      
         continue
        if len(c) < 100:           
         continue

        c = re.sub(r"\s+", " ", c)
        c = re.sub(r"[^a-zA-Z0-9 .,]", "", c)
        clean_chunks.append(c)
    print(f"Number of clean chunks: {len(clean_chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, embeddings, chunks_for_index = build_vectorstore(clean_chunks)
    print("Vector store created.")

    top_chunks_with_dist = search(query, model, index, chunks_for_index, k=top_k)
    top_chunks = [c for c, dist in top_chunks_with_dist if dist < 0.85]
    top_chunks_text = "\n\n".join(top_chunks)
    prompt = f"""
    You are a medical expert in urology.

    Using ONLY the information provided below,
    write a concise and clinically accurate definition
    for the following term.

    Term:
    {query}

    Context:
    {top_chunks_text}

    Definition:
    """

    definition = generate_definition(prompt)

    return definition

if __name__ == "__main__":
    query = "Benign prostatic hyperplasia"
    definition = rag_generate_definition(query)
    print(f"\nDefinition for '{query}':\n{definition}")
