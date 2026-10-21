"""List all files in the MSMARCO-XI dataset repo before downloading anything."""
from huggingface_hub import list_repo_files

files = list_repo_files("ai4bharat/MSMARCO-XI", repo_type="dataset")
for f in files:
    print(f)