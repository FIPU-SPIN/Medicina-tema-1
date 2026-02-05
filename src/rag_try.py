import json
from pdf_load import load_pdfs
from chunkovi import chunk_text
from retriever import retrieve_chunks
from vector_store import build_vectorstore, search
from sentence_transformers import SentenceTransformer
from hf_api_llm import generate_definition

docs = load_pdfs("data/raw")
print(f"Total PDF pages loaded: {len(docs)}")

all_chunks = []
for doc in docs:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

clean_chunks = []
for c in all_chunks:
    if c.count("|") > 5:
        continue
    if c.count("C1") > 2:
        continue
    if len(c) < 100:
        continue
    clean_chunks.append(c)

print(f"Clean chunks after filtering: {len(clean_chunks)}")
print("Example:")
for i, c in enumerate(clean_chunks[:3]):
    print(f"CHUNK {i+1}:\n{c[:1000]}\n{'-'*80}\n")

definition_chunks = []
for c in clean_chunks:
    if any(k in c.lower() for k in [" is ", " defined as ", " refers to "]):
        if 50 <= len(c.split()) <= 500:
            definition_chunks.append(c)

print(f"Chunkovi suitable for definition generation: {len(definition_chunks)}")
print("\nPrimjer chunk-a:\n")
if definition_chunks:
     print("\nExample definition chunk:\n")
     print(definition_chunks[0][:500])
     print("\n" + "-"*80)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index, embeddings, chunks_for_index = build_vectorstore(definition_chunks)
print(f"Vector store created with {len(chunks_for_index)} definition chunks")

queries = [
    "Benign prostatic hyperplasia",
    "Prostate cancer",
    "Cystoscopy",
    "Nephrolithiasis",
]

all_definitions = {}

for query in queries:
    top_chunks = retrieve_chunks(clean_chunks, query, top_k=5)
    if not top_chunks:
        print(f"No relevant chunks found for query: {query}")
        continue
    results = search(query, embedding_model, index, chunks_for_index, k=5)
    
    top_chunks_text = "\n\n".join(top_chunks)
  
    try:
        definition = generate_definition(top_chunks_text, query)
        all_definitions[query] = definition
        print(f"\nDefinition for '{query}':\n{definition}\n{'='*80}")
    except Exception as e:
        print(f"Error generating definition for '{query}': {e}")

    with open("data/processed/definitions.json", "w", encoding="utf-8") as f:
        json.dump(all_definitions, f, indent=2)

print(f"\nAll definitions saved to 'data/processed/definitions.json'")