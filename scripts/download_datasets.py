
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_url

COMMIT = "a8510cfedc491496ff4ea4ef2de6c9988387eb56"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def urls(task):
    if task=="JNLI":
        files = {"train":"JNLI/jglue-train.parquet","validation":"JNLI/jglue-validation.parquet"}
    elif task=="JSQuAD":
        files = {"train":"JSQuAD/jglue-train.parquet","validation":"JSQuAD/jglue-validation.parquet"}
    else:
        raise ValueError(task)
    return {k: hf_hub_url("shunk031/JGLUE", v, repo_type="dataset", revision=COMMIT) for k,v in files.items()}

def main():
    ensure_dir("data")
    for task in ["JNLI","JSQuAD"]:
        u = urls(task)
        print(f"Loading {task} from parquet (commit a8510cf) …")
        ds = load_dataset("parquet", data_files=u)
        for split, d in ds.items():
            print(f"  {task} | {split}: {len(d)} rows")
    print("Done.")

if __name__ == "__main__":
    main()
