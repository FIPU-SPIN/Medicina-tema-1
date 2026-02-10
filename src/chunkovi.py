def chunk_text(text, chunk_size=300, overlap=50):
    # safety check
    if not text:
        return []
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

        # ako idući korak nema dovoljno chunkova
        if i + overlap >= len(words) and len(words) > chunk_size:
            break
    return chunks
