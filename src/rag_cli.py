from pdf_load import load_pdfs
from chunkovi import chunk_text
from vector_store import build_vectorstore, search
from hf_api_llm import generate_definition
from sentence_transformers import SentenceTransformer

docs = load_pdfs("data/raw")
all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_text(doc))
clean_chunks = [c for c in all_chunks if len(c) > 100 and c.count("|") <= 5 and c.count("C1") <= 2]

model = SentenceTransformer("all-MiniLM-L6-v2")
index, embeddings, chunks_for_index = build_vectorstore(clean_chunks)

def rag_query(user_input, top_k=5):
    top_chunks = search(user_input, model, index, chunks_for_index)[:top_k]
    top_chunks_text = "\n\n".join(top_chunks)
    definition = generate_definition(top_chunks_text, user_input)
    return definition

if __name__ == "__main__":
    while True:
        user_input = input("Enter your medical query (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        definition = rag_query(user_input)
        print(f"\nDefinition for '{user_input}':\n{definition}\n{'='*60}\n")
