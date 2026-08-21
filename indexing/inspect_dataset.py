from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI")
print(ds)
print(ds["train"][0])
print(ds["train"].features)