from pdf_load import load_pdfs
from vector_store import build_vectorstore, search
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


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

keywords = ["benign", "prostatic", "hyperplasia"]

print("Kljucne rijeci:")

found = 0
for c in clean_chunks:
    cl = c.lower()
    if all(k in cl for k in keywords):
        print(c[:1000])
        print("\n" + "-" * 80 + "\n")
        found += 1
    if found == 3:
        break