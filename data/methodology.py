import pandas as pd
from pathlib import Path

evaluation_metrics_table = pd.DataFrame([
    {
        "Metric": "BERTScore Precision",
        "Underlying Model": "Transformer contextual embeddings",
        "Level of Analysis": "Token-level",
        "Mathematical Basis": "Cosine similarity of aligned token embeddings",
        "Evaluation Dimension": "Completeness",
        "Purpose": "Measures proportion of generated content aligned with reference"
    },
    {
        "Metric": "BERTScore Recall",
        "Underlying Model": "Transformer contextual embeddings",
        "Level of Analysis": "Token-level",
        "Mathematical Basis": "Cosine similarity of aligned token embeddings",
        "Evaluation Dimension": "Completeness",
        "Purpose": "Measures coverage of reference elements"
    },
    {
        "Metric": "BERTScore F1",
        "Underlying Model": "Transformer contextual embeddings",
        "Level of Analysis": "Token-level",
        "Mathematical Basis": "Harmonic mean of Precision and Recall",
        "Evaluation Dimension": "Clinical Correctness",
        "Purpose": "Primary correctness metric"
    },
    {
        "Metric": "Cosine Similarity (ClinicalBERT)",
        "Underlying Model": "Bio_ClinicalBERT",
        "Level of Analysis": "Sentence-level",
        "Mathematical Basis": "Cosine similarity between pooled sentence embeddings",
        "Evaluation Dimension": "Global Semantic Alignment",
        "Purpose": "Measures overall semantic similarity"
    },
    {
        "Metric": "Combined Score",
        "Underlying Model": "BERTScore F1 + ClinicalBERT Cosine",
        "Level of Analysis": "Instance-level",
        "Mathematical Basis": "Arithmetic mean",
        "Evaluation Dimension": "Overall Performance",
        "Purpose": "Integrated alignment metric"
    }
])

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

metrics_table_path = output_dir / "evaluation_metrics_framework.xlsx"
evaluation_metrics_table.to_excel(metrics_table_path, index=False)

print(f"Evaluation metrics table saved to: {metrics_table_path}")