"""Inspect English fields specifically, using the working iter_batches path."""
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",
    repo_type="dataset",
)

pf = pq.ParquetFile(path)
batch = next(pf.iter_batches(batch_size=5))
df = batch.to_pandas()

for i in range(3):
    print(f"\n--- Row {i} ---")
    print("query_id:", df.iloc[i]["query_id"])
    print("Eng_Query:", df.iloc[i]["Eng_Query"])
    print("Eng_Answer:", df.iloc[i]["Eng_Answer"][:200])
    print("English_passages count:", len(df.iloc[i]["passages"]["English_passages"]))
    print("First passage:", df.iloc[i]["passages"]["English_passages"][0][:200])
    print("is_selected:", df.iloc[i]["passages"]["is_selected"])