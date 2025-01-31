import heapq
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from tqdm.contrib import tenumerate
from transformers import PreTrainedTokenizer

from loaders.tools import get_cos_sim


class ReviewDataLoader:
    """A data loader class for handling review data with features for recommendation systems.

    This class processes review data and provides functionality for loading, tokenizing,
    and managing features related to user reviews, including text processing and feature extraction.

    Attributes:
        device (str): The device to use for computations ('cuda' or 'cpu')
        word_dict (WordDictionary): Dictionary for managing word vocabularies
        user_dict (EntityDictionary): Dictionary for managing user entities
        item_dict (EntityDictionary): Dictionary for managing item entities
        max_rating (float): Maximum rating value in the dataset
        min_rating (float): Minimum rating value in the dataset
        tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for text processing
        seq_len (int): Maximum sequence length for text
        feature_pos_set (Set[str]): Set of positive features
        feature_neg_set (Set[str]): Set of negative features
        feature_pair_set (Set[str]): Set of feature pairs
        train (List[Dict[Any, Any]]): Training dataset
        valid (List[Dict[Any, Any]]): Validation dataset
        test (List[Dict[Any, Any]]): Test dataset
        user_profile_embeds (Optional[torch.Tensor]): User profile embeddings
        item_profile_embeds (Optional[torch.Tensor]): Item profile embeddings
        n_features (int): Number of features
    """

    def __init__(
        self,
        reviews: Dict[Any, Any],
        seq_len: int = 15,
        max_vocab_size: int = 20000,
        idx2word: List[str] = ["<bos>", "<eos>", "<pad>", "<unk>"],
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        """Initialize the ReviewDataLoader.

        Args:
            reviews (Dict[Any, Any]): Dictionary containing review data
            seq_len (int, optional): Maximum sequence length. Defaults to 15.
            max_vocab_size (int, optional): Maximum vocabulary size. Defaults to 20000.
            idx2word (List[str], optional): Initial word index list. Defaults to ["<bos>", "<eos>", "<pad>", "<unk>"].
            tokenizer (Optional[PreTrainedTokenizer], optional): Tokenizer for text processing. Defaults to None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.word_dict = WordDictionary(idx2word)
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float("-inf")
        self.min_rating = float("inf")
        self.tokenizer = tokenizer
        self.initialize(reviews)
        self.seq_len = seq_len
        self.word_dict.keep_most_frequent(max_vocab_size)
        self.__unk = self.word_dict.word2idx["<unk>"]
        self.feature_pos_set: List[str] = []
        self.feature_neg_set: List[str] = []
        self.feature_pair_set: List[str] = []
        self.train, self.valid, self.test = self.load_data(reviews)
        self.user_profile_embeds: Optional[torch.Tensor] = None
        self.item_profile_embeds: Optional[torch.Tensor] = None
        self.n_features: int = 0

    def initialize(self, reviews: Dict[Any, Any]) -> None:
        """Initialize the data loader with review data.

        Args:
            reviews (Dict[Any, Any]): Dictionary containing review data
        """
        for review in reviews:
            self.user_dict.add_entity(review["user"])
            self.item_dict.add_entity(review["item"])

            rating = review["rating"]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

            (pos, neg, tem) = review["template"]
            self.word_dict.add_word(pos)
            self.word_dict.add_word(neg)

            if self.tokenizer is None:
                self.word_dict.add_sentence(tem)

    def load_data(
        self, reviews: Dict[Any, Any]
    ) -> Tuple[
        List[Dict[Any, Any]], List[Dict[Any, Any]], List[Dict[Any, Any]]
    ]:
        """Load and process the review data into train, validation, and test sets.

        Args:
            reviews (Dict[Any, Any]): Dictionary containing review data

        Returns:
            Tuple[List[Dict[Any, Any]], List[Dict[Any, Any]], List[Dict[Any, Any]]]:
                Tuple containing train, validation, and test datasets
        """
        train, valid, test = [], [], []
        for review in reviews:
            (pos, neg, tem) = review["template"]
            data = {
                "user": self.user_dict.entity2idx[review["user"]],
                "item": self.item_dict.entity2idx[review["item"]],
                "rating": review["rating"],
                "text": tem,
                "feature": (
                    self.word_dict.word2idx.get(pos, self.__unk)
                    if pos is not None
                    else -100
                ),
                "feature_neg": (
                    self.word_dict.word2idx.get(neg, self.__unk)
                    if neg is not None
                    else -100
                ),
                "pos_text": pos,
                "neg_text": neg,
            }
            if self.tokenizer is None:
                data.update({"textids": self.__seq2ids(tem)})
            else:
                tokens = self.tokenizer(tem)["input_ids"]
                decoded_text = self.tokenizer.decode(tokens)
                text = " ".join(decoded_text.split()[: self.seq_len])
                data.update({"text": text})

            self.feature_pos_set.append(pos)
            self.feature_neg_set.append(neg)
            self.feature_pair_set.append(f"{pos} {neg}")  # biterm

            if review["role"] == 0:
                train.append(data)
            elif review["role"] == 1:
                valid.append(data)
            else:
                test.append(data)

        self.feature_pos_set = set(self.feature_pos_set)
        self.feature_neg_set = set(self.feature_neg_set)
        self.feature_pair_set = set(self.feature_pair_set)

        return train, valid, test

    def __seq2ids(self, seq: str) -> List[int]:
        """Convert a sequence of words to their corresponding indices.

        Args:
            seq (str): Input sequence of words

        Returns:
            List[int]: List of word indices
        """
        return [
            self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()
        ]

    def __seq2ids_aspect(self, seq: List[str]) -> List[int]:
        """Convert a sequence of aspect words to their corresponding indices.

        Args:
            seq (List[str]): List of aspect words

        Returns:
            List[int]: List of aspect word indices, padded to length 4
        """
        if len(seq) == 0:
            return [self.__unk for _ in range(4)]
        if len(seq) == 1:
            ids = [
                self.word_dict.word2idx.get(w, self.__unk)
                for w in seq[0].split(" ")
            ][:2]
            return ids + [self.__unk, self.__unk]

        ids1 = [
            self.word_dict.word2idx.get(w, self.__unk)
            for w in seq[0].split(" ")
        ][:2]
        ids2 = [
            self.word_dict.word2idx.get(w, self.__unk)
            for w in seq[1].split(" ")
        ][:2]
        return ids1 + ids2

    def __prepare_search_data(self, encoder_name: str = "") -> Tuple[
        Optional[SentenceTransformer],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
        List[str],
        List[str],
        List[str],
    ]:
        """Prepare data for search operations.

        Args:
            encoder_name (str, optional): Name of the sentence transformer model. Defaults to "".

        Returns:
            Tuple containing:
                - Optional[SentenceTransformer]: Sentence transformer model
                - torch.Tensor: User indices for training
                - torch.Tensor: Item indices for training
                - torch.Tensor: Ratings for training
                - List[str]: Training texts
                - List[str]: Training features
                - List[str]: Training negative features
                - List[str]: Training feature pairs
        """
        model = None
        if encoder_name != "":
            model = SentenceTransformer(encoder_name).to(self.device)
        useridxs_train = []
        itemidxs_train = []
        ratings_train = []
        texts_train = []
        feats_train = []
        neg_train = []
        feat_pos_neg_train = []
        for data in self.train:
            useridxs_train.append(data["user"])
            itemidxs_train.append(data["item"])
            ratings_train.append(data["rating"])
            texts_train.append(data["text"])
            feats_train.append(data["pos_text"])
            neg_train.append(data["neg_text"])
            feat_pos_neg_train.append(f"{data['pos_text']} {data['neg_text']}")
        useridxs_train = torch.tensor(useridxs_train).to(self.device)
        itemidxs_train = torch.tensor(itemidxs_train).to(self.device)
        ratings_train = torch.tensor(ratings_train).to(self.device)

        return (
            model,
            useridxs_train,
            itemidxs_train,
            ratings_train,
            texts_train,
            feats_train,
            neg_train,
            feat_pos_neg_train,
        )

    def add_features_ui(self, n_features: int = 3, mode: int = 0) -> None:
        """Add user-item features to the dataset.

        Args:
            n_features (int, optional): Number of features to add. Defaults to 3.
            mode (int, optional): Feature mode (0=pos only, 1=neg only, 2=both). Defaults to 0.
        """
        (
            _,
            useridxs_train,
            itemidxs_train,
            _,
            _,
            feats_train,
            neg_train,
            feat_pos_neg_train,
        ) = self.__prepare_search_data()

        if mode == 0:
            stores = feats_train
        elif mode == 1:
            stores = neg_train
        elif mode == 2:
            stores = feat_pos_neg_train
        else:
            raise NotImplementedError

        for d in [self.train, self.valid, self.test]:
            self.__add_retrieved_features_ui(
                d,
                useridxs_train,
                itemidxs_train,
                stores,
                n_features,
            )

        self.n_features = n_features

    def __add_retrieved_features_ui(
        self,
        reviews: List[Dict[Any, Any]],
        useridxs_train: torch.Tensor,
        itemidxs_train: torch.Tensor,
        stores: List[str],
        n_features: int,
    ) -> None:
        """Add retrieved features to user-item pairs.

        Args:
            reviews (List[Dict[Any, Any]]): List of reviews to process
            useridxs_train (torch.Tensor): Training user indices
            itemidxs_train (torch.Tensor): Training item indices
            stores (List[str]): List of features to retrieve from
            n_features (int): Number of features to retrieve
        """
        for i, data in tenumerate(reviews, desc="Features Retrieval"):
            user, item = torch.tensor(data["user"]).to(
                self.device
            ), torch.tensor(data["item"]).to(self.device)

            # 1. Get all past features from the target user
            idxs_candidate = torch.where(useridxs_train == user)[0]
            feats = [
                stores[idx].split()
                for idx in idxs_candidate.tolist()
                if stores[idx] is not None
            ]
            feats_u = set([item for sub in feats for item in sub])

            # 2. Get all past features from the target item
            idxs_candidate = torch.where(itemidxs_train == item)[0]
            feats = [
                stores[idx].split()
                for idx in idxs_candidate.tolist()
                if stores[idx] is not None
            ]
            feats_i = set([item for sub in feats for item in sub])

            # 3. Merge and randomize
            feats_common = list(feats_u.intersection(feats_i))
            feats_unique = list(feats_u.symmetric_difference(feats_i))
            random.shuffle(feats_common)
            random.shuffle(feats_unique)
            # NOTE: Sometimes, it contains multiple words (previous dataset).
            feats = [fea.split()[-1] for fea in (feats_common + feats_unique)]

            # 4. Get the top-n_features
            reviews[i]["retrieved_feats_ui"] = " ".join(feats[:n_features])

            if i % 10000 == 0:
                print(
                    "[test] reviews[i]['retrieved_feats_ui']: ",
                    reviews[i]["retrieved_feats_ui"],
                )

    def add_aspects(
        self, encoder_name: str = "paraphrase-MiniLM-L6-v2"
    ) -> None:
        """Add aspects to the dataset using a sentence transformer model.

        Args:
            encoder_name (str, optional): Name of the sentence transformer model.
                Defaults to "paraphrase-MiniLM-L6-v2".
        """
        (
            model,
            useridxs_train,
            itemidxs_train,
            _,
            _,
            _,
            _,
            pos_neg_train,
        ) = self.__prepare_search_data(encoder_name)
        assert model is not None, "sentence encoder model should be included"
        embed_features_train = model.encode(
            pos_neg_train,
            batch_size=128,
            convert_to_tensor=True,
            device=self.device,
        )

        for d in [self.train, self.valid, self.test]:
            self.__add_retrieved_aspects(
                d,
                useridxs_train,
                itemidxs_train,
                pos_neg_train,
                embed_features_train,
            )

    def __add_retrieved_aspects(
        self,
        reviews: List[Dict[Any, Any]],
        useridxs_train: torch.Tensor,
        itemidxs_train: torch.Tensor,
        pos_neg_train: List[str],
        embed_features_train: torch.Tensor,
    ) -> None:
        """Add retrieved aspects to the reviews.

        Args:
            reviews (List[Dict[Any, Any]]): List of reviews to process
            useridxs_train (torch.Tensor): Training user indices
            itemidxs_train (torch.Tensor): Training item indices
            pos_neg_train (List[str]): List of positive-negative feature pairs
            embed_features_train (torch.Tensor): Embedded features for training
        """
        for i, data in tenumerate(reviews, desc="Aspects Retrieval"):
            user, item = torch.tensor(data["user"]).to(
                self.device
            ), torch.tensor(data["item"]).to(self.device)

            # 1. Get features (aspect candidates) of the target user
            idxs_candidate = torch.where(useridxs_train == user)[0]
            embed_candidate = embed_features_train[idxs_candidate]

            # 2. Get the average of feature vectors for the target item
            idxs_i = torch.where(itemidxs_train == item)[0]
            embed_avg_item = torch.mean(embed_features_train[idxs_i], dim=0)

            # 3. Get the top two idx of features with similar embeddings.
            scores = get_cos_sim(embed_avg_item, embed_candidate)
            sorted_idxs = torch.argsort(scores, descending=True)
            sorted_idxs = idxs_candidate[sorted_idxs[:2]]

            aspects = []
            if sorted_idxs.size(0) != 0:
                aspects = [pos_neg_train[idx] for idx in sorted_idxs.tolist()]
            reviews[i]["aspect"] = self.__seq2ids_aspect(aspects)

    def add_profiles(
        self,
        encoder_name: str = "paraphrase-MiniLM-L6-v2",
        n_texts: int = 3,
    ) -> None:
        """Add user and item profiles using a sentence transformer model.

        Args:
            encoder_name (str, optional): Name of the sentence transformer model.
                Defaults to "paraphrase-MiniLM-L6-v2".
            n_texts (int, optional): Number of texts to include in profiles. Defaults to 3.
        """
        (
            model,
            useridxs_train,
            itemidxs_train,
            _,
            texts_train,
            _,
            _,
            _,
        ) = self.__prepare_search_data(encoder_name)
        assert model is not None, "sentence encoder model is not defined"
        embed_exp_train = model.encode(
            texts_train,
            batch_size=128,
            convert_to_tensor=True,
            device=self.device,
        )
        user_profiles, item_profiles = self.__generate_profiles(
            useridxs_train,
            itemidxs_train,
            texts_train,
            embed_exp_train,
            n_texts,
        )
        self.user_profile_embeds = model.encode(
            user_profiles,
            batch_size=128,
            convert_to_tensor=True,
            device=self.device,
        )
        self.item_profile_embeds = model.encode(
            item_profiles,
            batch_size=128,
            convert_to_tensor=True,
            device=self.device,
        )

    def __generate_profiles(
        self,
        useridxs_train: torch.Tensor,
        itemidxs_train: torch.Tensor,
        texts_train: List[str],
        embed_exp_train: torch.Tensor,
        n_texts: int = 3,
    ) -> Tuple[List[str], List[str]]:
        """Generate user and item profiles from training data.

        Args:
            useridxs_train (torch.Tensor): Training user indices
            itemidxs_train (torch.Tensor): Training item indices
            texts_train (List[str]): Training texts
            embed_exp_train (torch.Tensor): Embedded training texts
            n_texts (int, optional): Number of texts to include in profiles. Defaults to 3.

        Returns:
            Tuple[List[str], List[str]]: Tuple containing user profiles and item profiles
        """
        user_profiles = []
        for i in tqdm(
            range(len(self.user_dict)), desc="Profile Retrieval (User)"
        ):
            # 1. Pool past reviews from the target user
            idxs_u = torch.where(useridxs_train == i)[0]
            embed_avg_u = torch.mean(embed_exp_train[idxs_u], dim=0)

            # 2. Retrieve similar reviews from all user reviews
            scores = get_cos_sim(embed_avg_u, embed_exp_train)
            sorted_idxs = torch.argsort(scores, descending=True)
            texts_profile = [
                texts_train[idx] for idx in sorted_idxs[:n_texts].tolist()
            ]
            user_profile = " ".join(texts_profile)
            user_profiles.append(user_profile)

        item_profiles = []
        for i in tqdm(
            range(len(self.item_dict)), desc="Profile Retrieval (Item)"
        ):
            # 1. Pool past reviews from the target item
            idxs_i = torch.where(itemidxs_train == i)[0]
            embed_avg_i = torch.mean(embed_exp_train[idxs_i], dim=0)

            # 2. Retrieve similar reviews from all item reviews
            scores = get_cos_sim(embed_avg_i, embed_exp_train)
            sorted_idxs = torch.argsort(scores, descending=True)
            texts_profile = [
                texts_train[idx] for idx in sorted_idxs[:n_texts].tolist()
            ]
            item_profile = " ".join(texts_profile)
            item_profiles.append(item_profile)

        return user_profiles, item_profiles

    def add_pred_rating(
        self,
        train_ratings: Optional[List[float]],
        valid_ratings: Optional[List[float]],
        test_ratings: Optional[List[float]],
        leak_rating: bool = False,
        noise_std: float = 0,
        min_rating: int = 1,
        max_rating: int = 5,
    ) -> Tuple[float, float]:
        """Add predicted ratings to the dataset.

        Args:
            train_ratings (Optional[List[float]]): Predicted ratings for training set
            valid_ratings (Optional[List[float]]): Predicted ratings for validation set
            test_ratings (Optional[List[float]]): Predicted ratings for test set
            leak_rating (bool, optional): Whether to leak true ratings. Defaults to False.
            noise_std (float, optional): Standard deviation of noise to add. Defaults to 0.
            min_rating (int, optional): Minimum rating value. Defaults to 1.
            max_rating (int, optional): Maximum rating value. Defaults to 5.

        Returns:
            Tuple[float, float]: Tuple containing MAE and RMSE metrics
        """
        errors = np.zeros(len(self.test))

        # (train)
        for d in self.train:
            d["pred_rating"] = d["rating"]

        # (valid)
        if leak_rating:
            for d in self.valid:
                d["pred_rating"] = d["rating"]
        else:
            for i, rating in enumerate(valid_ratings):
                self.valid[i]["pred_rating"] = rating

        # (test)
        if leak_rating:
            noise = np.zeros(len(self.test))
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, len(self.test))
                for i, (d, n) in enumerate(zip(self.test, noise)):
                    d["pred_rating"] = np.minimum(
                        max_rating, np.maximum(min_rating, d["rating"] + n)
                    )
                    errors[i] = d["rating"] - d["pred_rating"]
            else:
                for d in self.test:
                    d["pred_rating"] = d["rating"]
        else:
            for i, rating in enumerate(test_ratings):
                errors[i] = self.test[i]["rating"] - rating
                self.test[i]["pred_rating"] = rating

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        return mae, rmse


class WordDictionary:
    """A dictionary class for managing word vocabularies.

    Attributes:
        idx2word (List[str]): List of words indexed by their IDs
        word2idx (Dict[str, int]): Dictionary mapping words to their IDs
        __word2count (Dict[str, int]): Dictionary tracking word frequencies
        __num_predefine_words (int): Number of predefined special words
    """

    def __init__(
        self, idx2word: List[str] = ["<bos>", "<eos>", "<pad>", "<unk>"]
    ) -> None:
        """Initialize the WordDictionary.

        Args:
            idx2word (List[str], optional): Initial list of words.
                Defaults to ["<bos>", "<eos>", "<pad>", "<unk>"].
        """
        self.idx2word = idx2word
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.__word2count: Dict[str, int] = {}
        self.__num_predefine_words = len(self.idx2word)

    def add_sentence(self, sentence: str) -> None:
        """Add all words in a sentence to the dictionary.

        Args:
            sentence (str): Input sentence to process
        """
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """Add a single word to the dictionary.

        Args:
            word (str): Word to add
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.__word2count[word] = 1
        else:
            self.__word2count[word] += 1

    def __len__(self) -> int:
        """Get the size of the dictionary.

        Returns:
            int: Number of words in the dictionary
        """
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size: int = 20000) -> None:
        """Keep only the most frequent words up to max_vocab_size.

        Args:
            max_vocab_size (int, optional): Maximum vocabulary size. Defaults to 20000.
        """
        if len(self.__word2count) > max_vocab_size:
            freq_words = heapq.nlargest(
                max_vocab_size, self.__word2count, key=self.__word2count.get
            )
            self.idx2word = (
                freq_words + self.idx2word[: self.__num_predefine_words]
            )
            self.word2idx = {
                word: idx for idx, word in enumerate(self.idx2word)
            }


class EntityDictionary:
    """A dictionary class for managing entities (users/items).

    Attributes:
        entity2idx (Dict[Any, int]): Dictionary mapping entities to their IDs
        idx2entity (List[Any]): List of entities indexed by their IDs
    """

    def __init__(self) -> None:
        """Initialize the EntityDictionary."""
        self.entity2idx: Dict[Any, int] = {}
        self.idx2entity: List[Any] = []

    def add_entity(self, entity: Any) -> None:
        """Add an entity to the dictionary.

        Args:
            entity (Any): Entity to add
        """
        if entity not in self.entity2idx:
            self.entity2idx[entity] = len(self.idx2entity)
            self.idx2entity.append(entity)

    def __len__(self) -> int:
        """Get the size of the dictionary.

        Returns:
            int: Number of entities in the dictionary
        """
        return len(self.idx2entity)
