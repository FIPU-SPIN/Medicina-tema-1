# Ako nam ne treba, lako maknemo cijeli py file

import fitz
import json
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_data.append({
            "page": page_num + 1,
            "content": text.strip()
        })
    
    doc.close()
    return text_data

def extract_tables_as_dict(pdf_path):
    doc = fitz.open(pdf_path)
    tables = []

    for page in doc:
        tabs = page.find_tables()
        for i, tab in enumerate(tabs):
            tables.append({
                "page": page.number + 1,
                "table_index": i,
                "data": tab.to_pandas().to_dict()
            })
    
    doc.close()
    return tables

def main():
    file_path = "Campbell-Walsh Urology 12th Edition Review -- Alan J_ Wein.pdf"
    
    if not Path(file_path).exists():
        print(f"Error: File {file_path} nout found.")
        return

    print(f"Extraction start: {file_path} ---")
    
    content = extract_text_from_pdf(file_path)
    
    output_file = "ekstrakcija_rezultat.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Data saved in {output_file}")

if __name__ == "__main__":
    main()