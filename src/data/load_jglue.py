
from datasets import load_dataset
from huggingface_hub import hf_hub_url

COMMIT = "a8510cfedc491496ff4ea4ef2de6c9988387eb56"

def _urls(task):
    table = {
        "JNLI": {"train":"JNLI/jglue-train.parquet","validation":"JNLI/jglue-validation.parquet"},
        "JSQuAD": {"train":"JSQuAD/jglue-train.parquet","validation":"JSQuAD/jglue-validation.parquet"},
    }[task]
    return {k: hf_hub_url("shunk031/JGLUE", v, repo_type="dataset", revision=COMMIT) for k,v in table.items()}

def load_jnli(split="validation"):
    u = _urls("JNLI")[split]
    return load_dataset("parquet", data_files={split: u})[split]

def load_jsquad(split="validation"):
    u = _urls("JSQuAD")[split]
    return load_dataset("parquet", data_files={split: u})[split]
