import pandas as pd

df = pd.read_excel("data/processed/kbc_ri_data_ocisceno.xlsx")

questions = []

for idx, row in df.iterrows():
    param = row.get("parametar")
    value = row.get("vrijednost")
    unit = row.get("jedinica")

    question = f"Što znači vrijednost {param} od {value} {unit}?"
    
    questions.append({
        "id": idx,
        "parametar": param,
        "vrijednost": value,
        "pitanje": question
    })

q_df = pd.DataFrame(questions)
q_df.to_csv("data/processed/llm_pitanja.csv", index=False)

print("Generirana pitanja spremljena u data/processed/llm_pitanja.csv")