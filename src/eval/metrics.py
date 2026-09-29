"""Métriques d'évaluation style SQuAD"""
import re
import string
from collections import Counter


def normalize(s: str) -> str:
    """Normalisation SQuAD : minuscules, suppression ponctuation, articles, espaces."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    """Correspondance exacte sur formes normalisées."""
    return normalize(pred) == normalize(gold)


def f1(pred: str, gold: str) -> float:
    """F1 token-level sur formes normalisées."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    assert exact_match("The Arthur's Magazine", "Arthur's Magazine") is True
    assert exact_match("A cat.", "cat") is True
    assert f1("President Richard Nixon", "Richard Nixon") > 0.6
    assert f1("Paris", "London") == 0.0
    assert f1("", "Paris") == 0.0
    print("Tous les tests unitaires passent.")