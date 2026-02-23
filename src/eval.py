import os
import pandas as pd
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import torch

input_excel = 'data/raw/evaluacija_cisto.xlsx'  
output_excel = 'data/processed/model_outputs_cosine2.xlsx'

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

print("Calculating cosine similarity")

model_cbert = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
embeddings_gt = model_cbert.encode(df['ground_truth'].tolist(), convert_to_tensor=True)
embeddings_out = model_cbert.encode(df['output'].tolist(), convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings_out, embeddings_gt)  
cosine_scores_diag = cosine_scores.diag().cpu().numpy()

df['cosine_cBERT'] = cosine_scores_diag
df.to_excel(output_excel, index=False, engine='openpyxl')
print(f"cBERT with cosine similarity saved to {output_excel}")

df['combined_score'] = (df['bert_F1'] + df['cosine_cBERT']) / 2

model_avg = df.groupby('model_name').agg(
    avg_bert_F1=('bert_F1', 'mean'),
    avg_cosine_cBERT=('cosine_cBERT', 'mean'),
    avg_combined=('combined_score', 'mean')
).sort_values(by='avg_combined', ascending=False)

avg_output_excel = 'data/processed/model_avg_scores.xlsx'
model_avg.to_excel(avg_output_excel)
print(f"Average scores per model saved to {avg_output_excel}")
