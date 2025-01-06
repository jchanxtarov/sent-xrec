from itertools import chain
from typing import List, Tuple

import spacy
import torch

nlp = spacy.load("en_core_web_sm")


def get_cos_sim(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    if query.size() != key.size():
        query = query.unsqueeze(0)
    dot_product = torch.matmul(key, query.T).squeeze(1)
    norm_k = torch.norm(key, dim=1)
    norm_q = torch.norm(query, dim=1)
    return dot_product / (norm_k * norm_q)


def sentence_format(
    sentence: List[int], max_len: int, pad: int, bos: int, eos: int
) -> List[int]:
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


def get_pos_tags(words: List[str]) -> List[Tuple[str, str]]:
    doc = nlp(" ".join(words))
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags


def filter_words(words: List[str]) -> List[str]:
    pos_tags = get_pos_tags(words)
    filtered_words = {
        "nouns": [word for word, pos in pos_tags if pos == "NOUN"],
        "adjectives": [word for word, pos in pos_tags if pos == "ADJ"],
        "verbs": [word for word, pos in pos_tags if pos == "VERB"],
    }
    combined_list = list(chain(*filtered_words.values()))
    return combined_list
