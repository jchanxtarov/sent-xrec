import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import torch
from bert_score import score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def get_bleu_score(
    references: List[List[str]], hypothesis: List[List[str]], **kwargs
) -> dict:

    scores: Dict[str, float] = defaultdict(float)
    weights = [
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (0.333, 0.333, 0.333, 0.0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    types = ["bleu1", "bleu2", "bleu3", "bleu4"]

    # Calculate BLEU-1 and BLEU-4 score
    # Smoothing function is used to avoid division by zero
    # https://www.nltk.org/api/nltk.translate.bleu_score.html
    smoothing = SmoothingFunction().method1
    ref = [[[word.lower() for word in words]] for words in references]
    hyp = [[word.lower() for word in words] for words in hypothesis]

    for t, ws in zip(types, weights):
        scores[t] = corpus_bleu(
            list_of_references=ref,
            hypotheses=hyp,
            weights=ws,
            smoothing_function=smoothing,
            **kwargs,
        )

    return scores


def get_bert_score(
    references: List[str],
    hypothesis: List[str],
) -> dict:
    # see also: https://pypi.org/project/bert-score/
    # "en": roberta-large
    p, r, f1 = score(references, hypothesis, lang="en", verbose=False)

    p_mean = torch.mean(p).item()
    r_mean = torch.mean(r).item()
    f1_mean = torch.mean(f1).item()

    return {
        "bert-p": p_mean,
        "bert-r": r_mean,
        "bert-f": f1_mean,
    }


def get_rouge_score(
    references: List[str],
    hypothesis: List[str],
) -> dict:
    rouge_type = ["rouge1", "rouge2", "rougeL"]

    scores: Dict[str, float] = defaultdict(float)
    # Prepare the data for ROUGE calculation
    # ROUGE expects a list of strings for the references and hypothesis
    scorer = rouge_scorer.RougeScorer(rouge_type, use_stemmer=True)
    # Calculate ROUGE scores
    for ref, hyp in zip(references, hypothesis):
        score = scorer.score(target=ref.lower(), prediction=hyp.lower())
        for score_type in rouge_type:
            scores[score_type + "-p"] += score[score_type].precision
            scores[score_type + "-r"] += score[score_type].recall
            scores[score_type + "-f"] += score[score_type].fmeasure

    # Average ROUGE scores
    n = len(references)
    scores = {key: val / n for key, val in scores.items()}

    return scores


def is_two_sequence_same(sentence_a: List[str], sentence_b: List[str]) -> bool:
    if len(sentence_a) != len(sentence_b):
        return False
    for word_a, word_b in zip(sentence_a, sentence_b):
        if word_a != word_b:
            return False
    return True


def get_unique_sentence_ratio(sequence_batch: List[List[str]]) -> Tuple[float, float]:
    uniq_sequences: List[List[str]] = []
    for seq in sequence_batch:
        count = 0
        for u_seq in uniq_sequences:
            if is_two_sequence_same(seq, u_seq):
                count += 1
                break
        if count == 0:
            uniq_sequences.append(seq)

    return len(uniq_sequences) / len(sequence_batch), len(uniq_sequences)


def feature_detection(
    seq_batch: List[List[str]], feature_pos: List[str]
) -> List[Set[str]]:
    feature_batch = []
    for seq in seq_batch:
        feature_list = []
        for word in seq:
            if word in feature_pos:
                feature_list.append(word)
        feature_batch.append(set(feature_list))

    return feature_batch


def get_feature_matching_ratio(
    feature_batch: List[List[str]], test_feature: List[str]
) -> float:
    numerator = 0
    denominator = 0
    for feat_set, feat in zip(feature_batch, test_feature):
        if feat is not None:
            denominator += 1
            if feat in feat_set:
                numerator += 1

    if denominator == 0:
        return 0

    return numerator / denominator


def get_feature_coverage_ratio(
    feature_batch: List[str], feature_pos: List[str]
) -> float:
    feature_set: Set[str] = set()
    for feat in feature_batch:
        feature_set = feature_set | feat

    return len(feature_set) / len(feature_pos)


def get_feature_diversity(feature_batch: List[Set[str]]) -> float:
    n = len(feature_batch)
    numerator = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, n):
            y = feature_batch[j]
            numerator += len(x & y)
    denominator = n * (n - 1) / 2
    if denominator == 0:
        return 0

    return numerator / denominator


def is_all_words_in_string(words: List[str], string: str) -> bool:
    return all(word in string for word in words)


def get_mean_absolute_error(
    predicted: Tuple[List[float], List[float]],
    max_rating: float,
    min_rating: float,
    is_mae: bool = True,
) -> float:
    total = 0.0
    for r, p in predicted:
        if p > max_rating:
            p = max_rating
        if p < min_rating:
            p = min_rating

        sub = p - r
        if is_mae:
            total += abs(sub)
        else:
            total += sub**2

    return total / len(predicted)


def get_root_mean_square_error(
    predicted: Tuple[List[float], List[float]], max_rating: float, min_rating: float
) -> float:
    mse = get_mean_absolute_error(predicted, max_rating, min_rating, False)
    return math.sqrt(mse)
