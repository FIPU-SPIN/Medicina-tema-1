from pdf_load import load_pdfs
from chunkovi import chunk_text
from retriever import retrieve_chunks
from vector_store import build_vectorstore, search
from sentence_transformers import SentenceTransformer
from openai import OpenAI

docs = load_pdfs("data/raw")
print("Ukupan broj tekstualnih blokova (PDF stranica):", len(docs))

all_chunks = []
for doc in docs:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print("Ukupan broj chunkova:", len(all_chunks))

clean_chunks = []

for c in all_chunks:
    if c.count("|") > 5:
        continue
    if c.count("C1") > 2:
        continue
    if len(c) < 100:
        continue
    clean_chunks.append(c)

print(f"Clean chunkovi: {len(clean_chunks)}")

print("Primjer clean chunkova:")
for i, c in enumerate(clean_chunks[:3]):
    print(f"CHUNK {i+1}:\n")
    print(c[:1000])
    print("\n" + "-" * 80 + "\n")

definition_chunks = []

for c in clean_chunks:
    if any(k in c.lower() for k in [" is ", " defined as ", " refers to "]):
        if 50 <= len(c.split()) <= 500:
            definition_chunks.append(c)

print(f"Chunkovi spremni za definicije: {len(definition_chunks)}")
print("\nPrimjer chunk-a:\n")
if definition_chunks:
    print(definition_chunks[0][:500])
    print("\n" + "-"*80)

model = SentenceTransformer("all-MiniLM-L6-v2")

index, embeddings, chunks_for_index = build_vectorstore(definition_chunks)
print("Vector store kreiran sa def chunkovima:", len(chunks_for_index))

query = "What is benign prostatic hyperplasia?"
top_chunks = retrieve_chunks(clean_chunks, query, top_k=5)
results = search(query, model, index, chunks_for_index)

print("\nNajrelevantniji chunkovi za upit:\n")
for r in results[:5]:
    print(r[:500]) 
    print("-"*80)


client = OpenAI(api_key="YOUR_KEY_HERE")

top_chunks_text = "\n\n".join(top_chunks)

prompt = f"Please summarize the following medical text into a concise definition:\n{top_chunks_text}"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

print("Definicija LLM-a:")
print(response.choices[0].message.content)

