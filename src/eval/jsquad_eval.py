import argparse, json, math
from rank_bm25 import BM25Okapi
from src.data.load_jglue import load_jsquad
from src.eval.metrics import em, f1

def split_sentences(text: str):
    # simple JP/EN/punct-based split fallback
    import re
    sents = re.split(r'(。|！|？|\n)', text)
    # recombine delimiter tokens
    res = []
    for i in range(0, len(sents)-1, 2):
        res.append(sents[i] + sents[i+1])
    if len(sents) % 2 == 1:
        res.append(sents[-1])
    return [s.strip() for s in res if s.strip()]

def bm25_select(context, question, topk=1):
    sents = split_sentences(context)
    tokenized = [list(s) for s in sents]  # char-level tokenization as fallback
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(question))
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return sents[idx]

def evaluate(split='validation'):
    ds = load_jsquad(split=split)
    total_em, total_f1, n = 0.0, 0.0, 0
    for ex in ds:
        ctx = ex['context']
        q = ex['question']
        gold = ex['answers']['text'][0] if ex['answers']['text'] else ''
        pred_sent = bm25_select(ctx, q, topk=1)
        # heuristic: choose longest common substring between pred_sent and question tokens union
        pred = pred_sent.strip()
        total_em += em(pred, gold)
        total_f1 += f1(pred, gold)
        n += 1
        if n % 500 == 0:
            print(f"Processed {n} examples...")
    print(json.dumps({'split': split, 'size': n, 'EM': total_em/n, 'F1': total_f1/n}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()
    evaluate(args.split)
