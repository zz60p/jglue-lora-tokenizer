from typing import List, Tuple
import re

def normalize_text(s: str) -> str:
    # basic normalization for EM/F1
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def em(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = min(p_tokens.count(t), g_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def accuracy(preds: List[int], golds: List[int]) -> float:
    correct = sum(int(p==g) for p,g in zip(preds,golds))
    return correct/len(golds) if golds else 0.0
