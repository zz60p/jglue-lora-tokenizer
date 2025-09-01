from fugashi import Tagger
from sudachipy import dictionary, tokenizer as tknz
from transformers import AutoTokenizer
import random
from src.data.load_jglue import load_jnli

tagger = Tagger()
sud = dictionary.Dictionary().create()
spm = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

def mecab_tokens(t): return [m.surface for m in tagger(t)]
def sudachi_tokens(t, mode=tknz.Tokenizer.SplitMode.C): return [m.surface() for m in sud.tokenize(t, mode)]
def spm_tokens(t): return spm.convert_ids_to_tokens(spm(t, add_special_tokens=False)["input_ids"])

def main():
    ds = load_jnli(split="validation")
    for i in random.sample(range(len(ds)), 5):
        s = ds[i]["sentence1"]
        print("TEXT:", s)
        print("MeCab:", mecab_tokens(s)[:50])
        print("Sudachi:", sudachi_tokens(s)[:50])
        print("SPM:", spm_tokens(s)[:50])
        print("-"*60)
    # 简单统计：三种方案的 token 平均长度（字符数/词数）
    import numpy as np
    sents = [x["sentence1"] for x in ds.select(range(500))]
    def avg_len(fn): 
        lens = [len(fn(s)) for s in sents]
        return float(np.mean(lens)), float(np.percentile(lens,95))
    print("Avg tokens (500 samples):")
    print("MeCab mean,p95:", avg_len(mecab_tokens))
    print("Sudachi mean,p95:", avg_len(sudachi_tokens))
    print("SPM mean,p95:", avg_len(spm_tokens))

if __name__=="__main__":
    main()
