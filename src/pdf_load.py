import os
from PyPDF2 import PdfReader

def load_pdfs(path):
    documents = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(path, file)
            reader = PdfReader(pdf_path)

    for page in reader.pages:
                text = page.extract_text()
                if text:
                    cleaned = " ".join(text.split())
                    documents.append(cleaned)
    return documents