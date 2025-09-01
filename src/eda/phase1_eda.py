import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.load_jglue import load_jnli, load_jsquad

OUT = Path("reports/phase1"); OUT.mkdir(parents=True, exist_ok=True)

def jnli_stats():
    rows=[]
    for split in ["train","validation"]:
        ds = load_jnli(split=split)
        labels = [int(x["label"]) for x in ds]
        lens1 = [len(x["sentence1"]) for x in ds]
        lens2 = [len(x["sentence2"]) for x in ds]
        row = dict(
            split=split, n=len(ds),
            lbl0=sum(1 for z in labels if z==0),
            lbl1=sum(1 for z in labels if z==1),
            lbl2=sum(1 for z in labels if z==2),
            s1_len_mean=float(np.mean(lens1)), s1_len_p95=float(np.percentile(lens1,95)),
            s2_len_mean=float(np.mean(lens2)), s2_len_p95=float(np.percentile(lens2,95)),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"jnli_stats.csv", index=False)
    # 图：sentence1/2 长度分布（validation）
    v = load_jnli(split="validation")
    s1 = [len(x["sentence1"]) for x in v]; s2 = [len(x["sentence2"]) for x in v]
    plt.figure(); plt.hist(s1, bins=50); plt.title("JNLI sentence1 length (val)"); plt.savefig(OUT/"jnli_s1_len_hist.png"); plt.close()
    plt.figure(); plt.hist(s2, bins=50); plt.title("JNLI sentence2 length (val)"); plt.savefig(OUT/"jnli_s2_len_hist.png"); plt.close()

def jsquad_stats():
    rows=[]
    for split in ["train","validation"]:
        ds = load_jsquad(split=split)
        clen = [len(x["context"]) for x in ds]
        qlen = [len(x["question"]) for x in ds]
        alen = [len(x["answers"]["text"][0]) if x["answers"]["text"] else 0 for x in ds]
        row = dict(
            split=split, n=len(ds),
            ctx_len_mean=float(np.mean(clen)), ctx_len_p95=float(np.percentile(clen,95)),
            q_len_mean=float(np.mean(qlen)), q_len_p95=float(np.percentile(qlen,95)),
            ans_len_mean=float(np.mean(alen)), ans_len_p95=float(np.percentile(alen,95)),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"jsquad_stats.csv", index=False)
    # 图：context 与 answer 长度直方图（validation）
    v = load_jsquad(split="validation")
    clen = [len(x["context"]) for x in v]
    alen = [len(x["answers"]["text"][0]) if x["answers"]["text"] else 0 for x in v]
    plt.figure(); plt.hist(clen, bins=60); plt.title("JSQuAD context length (val)"); plt.savefig(OUT/"jsquad_ctx_len_hist.png"); plt.close()
    plt.figure(); plt.hist(alen, bins=60); plt.title("JSQuAD answer length (val)"); plt.savefig(OUT/"jsquad_ans_len_hist.png"); plt.close()

if __name__ == "__main__":
    jnli_stats(); jsquad_stats()
    print("EDA artifacts written to reports/phase1/")
