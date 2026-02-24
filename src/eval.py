import os
import pandas as pd
from bert_score import score
from sentence_transformers import SentenceTransformer, util

def get_next_filename(base_path, base_name):
    version = 1
    while True:
        filename = f"{base_name}_{version}.xlsx"
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            return full_path
        version += 1

input_excel = 'data/raw/evaluacija_cisto.xlsx'  
output_excel = get_next_filename('data/processed', 'cosine')

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

avg_output_excel = get_next_filename('data/processed', 'model_avg_scores')
model_avg.to_excel(avg_output_excel)
print(f"Average scores per model saved to {avg_output_excel}")

df['combined_score'] = (df['bert_F1'] + df['cosine_cBERT']) / 2

model_stats = df.groupby('model_name').agg(
    avg_bert_F1=('bert_F1', 'mean'),
    std_bert_F1=('bert_F1', 'std'),
    avg_cosine_cBERT=('cosine_cBERT', 'mean'),
    std_cosine_cBERT=('cosine_cBERT', 'std'),
    avg_combined=('combined_score', 'mean'),
    std_combined=('combined_score', 'std')
).sort_values(by='avg_combined', ascending=False)

stats_output = get_next_filename('data/processed', 'statistics')
model_stats.to_excel(stats_output)

print(f"Statistics saved to {stats_output}")

correlation_matrix = df[['bert_F1', 'cosine_cBERT', 'combined_score']].corr(method='pearson')

corr_output = get_next_filename('data/processed', 'correlations')
correlation_matrix.to_excel(corr_output)

print(f"Correlation matrix saved to {corr_output}")

