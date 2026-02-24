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

"""
# =====================================================
# DODATNE STATISTIKE, IQR I ROBUST RANGIRANJE
# =====================================================

print("\nDodajem dodatne statistike, IQR i robust ranking...")

analiza_df = df.copy()

# -----------------------
# MEDIAN + IQR PO MODELU
# -----------------------

iqr_stats = analiza_df.groupby('model_name').agg(
    median_bert_F1=('bert_F1', 'median'),
    iqr_bert_F1=('bert_F1', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    median_cosine_cBERT=('cosine_cBERT', 'median'),
    iqr_cosine_cBERT=('cosine_cBERT', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    median_combined=('combined_score', 'median'),
    iqr_combined=('combined_score', lambda x: x.quantile(0.75) - x.quantile(0.25))
).round(4)

# -----------------------
# KLASIČNI RANKING (MEAN)
# -----------------------

ranking_mean = analiza_df.groupby('model_name').agg(
    avg_bert_F1=('bert_F1', 'mean'),
    avg_cosine_cBERT=('cosine_cBERT', 'mean'),
    avg_combined=('combined_score', 'mean')
).round(4)

ranking_mean['rank_bert_F1'] = ranking_mean['avg_bert_F1'].rank(ascending=False)
ranking_mean['rank_cosine_cBERT'] = ranking_mean['avg_cosine_cBERT'].rank(ascending=False)
ranking_mean['rank_combined'] = ranking_mean['avg_combined'].rank(ascending=False)

ranking_mean['overall_rank_mean'] = (
    ranking_mean['rank_bert_F1'] +
    ranking_mean['rank_cosine_cBERT'] +
    ranking_mean['rank_combined']
) / 3

ranking_mean = ranking_mean.sort_values('overall_rank_mean')

# -----------------------
# ROBUST RANKING (MEDIAN)
# -----------------------

ranking_median = analiza_df.groupby('model_name').agg(
    median_bert_F1=('bert_F1', 'median'),
    median_cosine_cBERT=('cosine_cBERT', 'median'),
    median_combined=('combined_score', 'median')
).round(4)

ranking_median['rank_bert_F1'] = ranking_median['median_bert_F1'].rank(ascending=False)
ranking_median['rank_cosine_cBERT'] = ranking_median['median_cosine_cBERT'].rank(ascending=False)
ranking_median['rank_combined'] = ranking_median['median_combined'].rank(ascending=False)

ranking_median['overall_rank_median'] = (
    ranking_median['rank_bert_F1'] +
    ranking_median['rank_cosine_cBERT'] +
    ranking_median['rank_combined']
) / 3

ranking_median = ranking_median.sort_values('overall_rank_median')

# -----------------------
# BROJ PRIMJERA
# -----------------------

sample_count = analiza_df.groupby('model_name').size().reset_index(name='num_samples')

# -----------------------
# GLOBALNA USPOREDBA
# -----------------------

metric_comparison = pd.DataFrame({
    'metric': ['bert_F1', 'cosine_cBERT', 'combined_score'],
    'mean': [
        analiza_df['bert_F1'].mean(),
        analiza_df['cosine_cBERT'].mean(),
        analiza_df['combined_score'].mean()
    ],
    'median': [
        analiza_df['bert_F1'].median(),
        analiza_df['cosine_cBERT'].median(),
        analiza_df['combined_score'].median()
    ],
    'iqr': [
        analiza_df['bert_F1'].quantile(0.75) - analiza_df['bert_F1'].quantile(0.25),
        analiza_df['cosine_cBERT'].quantile(0.75) - analiza_df['cosine_cBERT'].quantile(0.25),
        analiza_df['combined_score'].quantile(0.75) - analiza_df['combined_score'].quantile(0.25)
    ],
    'min': [
        analiza_df['bert_F1'].min(),
        analiza_df['cosine_cBERT'].min(),
        analiza_df['combined_score'].min()
    ],
    'max': [
        analiza_df['bert_F1'].max(),
        analiza_df['cosine_cBERT'].max(),
        analiza_df['combined_score'].max()
    ]
}).round(4)

# -----------------------
# SPREMANJE
# -----------------------

from pathlib import Path

output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

additional_stats_path = get_next_filename('data/processed', 'additional_model_analysis')

with pd.ExcelWriter(additional_stats_path, engine='openpyxl') as writer:
    iqr_stats.to_excel(writer, sheet_name='Median_and_IQR')
    ranking_mean.to_excel(writer, sheet_name='Ranking_mean')
    ranking_median.to_excel(writer, sheet_name='Robust_ranking_median')
    sample_count.to_excel(writer, sheet_name='Sample_count', index=False)
    metric_comparison.to_excel(writer, sheet_name='Metric_comparison', index=False)

print(f"Dodatne statistike spremljene u: {additional_stats_path}")
print("  - Median + IQR")
print("  - Klasični ranking (mean)")
print("  - Robust ranking (median)")
print("  - Broj primjera")
print("  - Globalna usporedba metrika")
"""