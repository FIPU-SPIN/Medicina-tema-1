def retrieve_chunks(chunks, query, top_k=3):
    query = query.lower()

    scored = []
    for chunk in chunks:
        score = chunk.lower().count(query)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_k]]
