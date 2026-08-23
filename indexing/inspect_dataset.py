"""Inspect MSMARCO-XI schema directly via pandas, bypassing datasets' builder.

Run list_files.py first to confirm the exact filename to use below.
"""
import pandas as pd
from huggingface_hub import hf_hub_download

# TODO: replace filename once list_files.py output confirms exact path
path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",  # placeholder — update after checking file list
    repo_type="dataset",
)

df = pd.read_parquet(path)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst row:\n", df.iloc[0])
print("\nDtypes:\n", df.dtypes)