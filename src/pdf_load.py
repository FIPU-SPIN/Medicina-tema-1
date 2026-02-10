import os
from PyPDF2 import PdfReader

def load_pdfs(path):
    documents = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(path, file)
            try:
                reader = PdfReader(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                if text:
                    cleaned = " ".join(text.split())
                    documents.append({
                        "source": "file",
                        "page": i+1,
                        "content": cleaned
                    })
            except Exception as e:
                print("Error while reading file {file}: {e}")
    return documents