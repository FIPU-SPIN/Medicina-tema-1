import os
import pandas as pd
from bert_score import score

input_excel = 'data/raw/evaluacija_cisto.xlsx'  
output_excel = 'data/processed/model_outputs.xlsx'

os.makedirs(os.path.dirname(output_excel), exist_ok=True)

df = pd.read_excel(input_excel)

print(df.head())
print(df.columns) 
df['output'] = df['output'].fillna('').astype(str)
df['ground_truth'] = df['ground_truth'].fillna('').astype(str)

candidates = df['output'].tolist()
references = df['ground_truth'].tolist()

print("Calculating")
P, R, F1 = score(candidates, references, lang='en')

df['bert_P'] = P.tolist()
df['bert_R'] = R.tolist()
df['bert_F1'] = F1.tolist()

df.to_excel(output_excel, index=False)
print(f"Metrics saved to {output_excel}")

avg_per_model = df.groupby('model_name')['bert_F1'].mean().reset_index()
avg_model_excel = 'data/processed/model_avg_F1.xlsx'
avg_per_model.to_excel(avg_model_excel, index=False)
print(f"Average F1 per model saved to {avg_model_excel}")
print(avg_per_model)