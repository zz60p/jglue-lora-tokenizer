# scripts/download_datasets.py  —— 无脚本版（直接读 Parquet）
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_url

REV = "refs/convert/parquet"  # 专门放 parquet/duckdb 的分支

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parquet_urls(task):
    if task == "JNLI":
        return {
            "train":      hf_hub_url("shunk031/JGLUE", "JNLI/jglue-train.parquet",      repo_type="dataset", revision=REV),
            "validation": hf_hub_url("shunk031/JGLUE", "JNLI/jglue-validation.parquet", repo_type="dataset", revision=REV),
        }
    if task == "JSQuAD":
        return {
            "train":      hf_hub_url("shunk031/JGLUE", "JSQuAD/jglue-train.parquet",      repo_type="dataset", revision=REV),
            "validation": hf_hub_url("shunk031/JGLUE", "JSQuAD/jglue-validation.parquet", repo_type="dataset", revision=REV),
        }
    raise ValueError(task)

def main():
    ensure_dir("data")
    for task in ["JNLI", "JSQuAD"]:
        urls = parquet_urls(task)
        print(f"Loading {task} from parquet …")
        ds = load_dataset("parquet", data_files=urls)   # 不执行任何远程代码
        for split, d in ds.items():
            print(f"  {task} | {split}: {len(d)} rows")
    print("Done.")

if __name__ == "__main__":
    main()
