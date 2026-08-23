"""Inspect MSMARCO-XI schema directly via pandas, bypassing datasets' builder."""
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",
    repo_type="dataset",
)

df = pd.read_parquet(path)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst row:\n", df.iloc[0])
print("\nDtypes:\n", df.dtypes)