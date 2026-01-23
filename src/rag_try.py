from pdf_load import load_pdfs

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

all_chunks = []

for doc in docs:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print("Ukupan broj chunkova:", len(all_chunks))

print("Ukupan broj tekstualnih blokova:", len(docs))
print("\nPrimjer:\n")
print(docs[0][:1000])

keywords = ["prostatic", "hyperplasia"]

print("\nRezultati:\n")

found = 0

for chunk in all_chunks:
    chunk_lower = chunk.lower()
    if all(k in chunk_lower for k in keywords):
        print(chunk)
        print("\n" + "-"*80 + "\n")
        found += 1

    if found == 3:
        break

