import argparse, json, os
from collections import Counter
from src.data.load_jglue import load_jnli
from src.eval.metrics import accuracy

def majority_label(train_split='train'):
    ds = load_jnli(split=train_split)
    cnt = Counter(int(x['label']) for x in ds)
    return cnt.most_common(1)[0][0]

def lexical_overlap_rule(premise, hypothesis):
    # very naive: if hypothesis tokens mostly appear in premise -> entailment, else neutral
    p = set(premise.split())
    h = set(hypothesis.split())
    overlap = len(p & h) / max(1, len(h))
    if overlap >= 0.6:
        return 0  # entailment (JGLUE: 0=entailment,1=contradiction,2=neutral per dataset card)
    return 2  # neutral

def evaluate(split='validation'):
    ds = load_jnli(split=split)
    maj = majority_label()
    preds_maj = [maj]*len(ds)
    acc_maj = accuracy(preds_maj, [int(x['label']) for x in ds])

    preds_rule = [lexical_overlap_rule(x['sentence1'], x['sentence2']) for x in ds]
    acc_rule = accuracy(preds_rule, [int(x['label']) for x in ds])

    print(json.dumps({
        'split': split,
        'size': len(ds),
        'majority_label': int(maj),
        'accuracy_majority': acc_maj,
        'accuracy_lexical_rule': acc_rule,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()
    evaluate(args.split)
