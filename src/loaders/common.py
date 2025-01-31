from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from loaders.tools import sentence_format


class XRecDataModule(pl.LightningDataModule):
    """A PyTorch Lightning data module for handling recommendation system data.

    This module manages the loading and preparation of data for training, validation,
    and testing in a recommendation system context.

    Attributes:
        batch_size (int): Size of batches for data loading
        max_seq_len (int): Maximum sequence length for text data
        storage (ReviewDataLoader): Data storage and processing handler
        bos_token (str): Beginning of sequence token
        eos_token (str): End of sequence token
        pad_token (str): Padding token
        tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for text processing
        is_recommender (bool): Whether this is a recommender system
        train_dataset (Union[XRecDataset, XRecTokenizerDataset]): Training dataset
        valid_dataset (Union[XRecDataset, XRecTokenizerDataset]): Validation dataset
        test_dataset (Union[XRecDataset, XRecTokenizerDataset]): Test dataset
    """

    def __init__(
        self,
        reviews: Dict[Any, Any],
        batch_size: int,
        max_seq_len: int,
        max_vocab_size: int,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pod>",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        is_recommender: bool = False,
    ) -> None:
        """Initialize the XRecDataModule.

        Args:
            reviews (Dict[Any, Any]): Dictionary containing review data
            batch_size (int): Size of batches for data loading
            max_seq_len (int): Maximum sequence length for text data
            max_vocab_size (int): Maximum vocabulary size
            bos_token (str, optional): Beginning of sequence token. Defaults to "<bos>".
            eos_token (str, optional): End of sequence token. Defaults to "<eos>".
            pad_token (str, optional): Padding token. Defaults to "<pod>".
            tokenizer (Optional[PreTrainedTokenizer], optional): Tokenizer for text processing. Defaults to None.
            is_recommender (bool, optional): Whether this is a recommender system. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.storage = ReviewDataLoader(
            reviews=reviews,
            seq_len=max_seq_len,
            max_vocab_size=max_vocab_size,
            tokenizer=tokenizer,
            idx2word=[bos_token, eos_token, pad_token, "<unk>"],
        )
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.tokenizer = tokenizer
        self.is_recommender = is_recommender

    def setup(self, stage: str) -> None:
        """Set up the datasets for training, validation, and testing.

        Args:
            stage (str): Stage of training ('fit', 'validate', 'test', or 'predict')
        """
        if self.tokenizer is not None:
            self.train_dataset = XRecTokenizerDataset(
                data=self.storage.train,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                tokenizer=self.tokenizer,
                # max_seq_len=self.max_seq_len, # Not used
                n_features=self.storage.n_features,
                # n_profs=self.storage.n_retrieved_profs, # Not used
                is_recommender=self.is_recommender,
            )
            self.valid_dataset = XRecTokenizerDataset(
                data=self.storage.valid,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                tokenizer=self.tokenizer,
                # max_seq_len=self.max_seq_len, # Not used
                n_features=self.storage.n_features,
                # n_profs=self.storage.n_retrieved_profs, # Not used
                is_recommender=self.is_recommender,
            )
            self.test_dataset = XRecTokenizerDataset(
                data=self.storage.test,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                tokenizer=self.tokenizer,
                # max_seq_len=self.max_seq_len, # Not used
                n_features=self.storage.n_features,
                # n_profs=self.storage.n_retrieved_profs, # Not used
                is_recommender=self.is_recommender,
            )
        else:
            self.train_dataset = XRecDataset(
                data=self.storage.train,
                bos_idx=self.storage.word_dict.word2idx[self.bos_token],
                eos_idx=self.storage.word_dict.word2idx[self.eos_token],
                pad_idx=self.storage.word_dict.word2idx[self.pad_token],
                max_seq_len=self.max_seq_len,
                is_recommender=self.is_recommender,
            )
            self.valid_dataset = XRecDataset(
                data=self.storage.valid,
                bos_idx=self.storage.word_dict.word2idx[self.bos_token],
                eos_idx=self.storage.word_dict.word2idx[self.eos_token],
                pad_idx=self.storage.word_dict.word2idx[self.pad_token],
                max_seq_len=self.max_seq_len,
                is_recommender=self.is_recommender,
            )
            self.test_dataset = XRecDataset(
                data=self.storage.test,
                bos_idx=self.storage.word_dict.word2idx[self.bos_token],
                eos_idx=self.storage.word_dict.word2idx[self.eos_token],
                pad_idx=self.storage.word_dict.word2idx[self.pad_token],
                max_seq_len=self.max_seq_len,
                is_recommender=self.is_recommender,
            )

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader.

        Returns:
            DataLoader: DataLoader for training data
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader.

        Returns:
            DataLoader: DataLoader for validation data
        """
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader.

        Returns:
            DataLoader: DataLoader for test data
        """
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def predict_dataloader(self) -> Dict[str, DataLoader]:
        """Get the prediction data loaders.

        Returns:
            Dict[str, DataLoader]: Dictionary containing DataLoaders for training and validation data
        """
        return {
            "train": DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
            "valid": DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
        }


class XRecDataset(Dataset):
    """Dataset class for recommendation system data without tokenizer.

    Attributes:
        is_recommender (bool): Whether this is a recommender system
        use_aspect (bool): Whether to use aspect information
        use_pred_rating (bool): Whether to use predicted ratings
        user (torch.Tensor): User indices
        item (torch.Tensor): Item indices
        rating (torch.Tensor): Rating values
        seq (torch.Tensor): Sequence data
        feature (torch.Tensor): Feature data
        feature_neg (torch.Tensor): Negative feature data
        aspect (Optional[torch.Tensor]): Aspect data
        pred_rating (Optional[torch.Tensor]): Predicted rating data
    """

    def __init__(
        self,
        data: List[Dict[Any, Any]],
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        max_seq_len: int = 15,
        is_recommender: bool = False,
    ) -> None:
        """Initialize the XRecDataset.

        Args:
            data (List[Dict[Any, Any]]): List of data samples
            bos_idx (int): Beginning of sequence index
            eos_idx (int): End of sequence index
            pad_idx (int): Padding index
            max_seq_len (int, optional): Maximum sequence length. Defaults to 15.
            is_recommender (bool, optional): Whether this is a recommender system. Defaults to False.
        """
        self.is_recommender = is_recommender
        self.use_aspect = False
        self.use_pred_rating = False

        u, i, r, t, f, fa, a, pr = [], [], [], [], [], [], [], []
        for x in data:
            u.append(x["user"])
            i.append(x["item"])
            r.append(x["rating"])
            t.append(
                sentence_format(
                    sentence=x["textids"],
                    max_len=max_seq_len,
                    pad=pad_idx,
                    bos=bos_idx,
                    eos=eos_idx,
                )
            )
            f.append([x["feature"]])
            fa.append([x["feature_neg"]])

            if "aspect" in x:  # erra
                self.use_aspect = True
                a.append(x["aspect"])

            if "pred_rating" in x:  # peter
                self.use_pred_rating = True
                pr.append(x["pred_rating"])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.feature_neg = torch.tensor(fa, dtype=torch.int64).contiguous()

        if self.use_aspect:
            self.aspect = torch.tensor(a, dtype=torch.int64).contiguous()

        if self.use_pred_rating:
            self.pred_rating = torch.tensor(pr, dtype=torch.float).contiguous()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self.user.size(0)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Union[torch.Tensor, float],
        ],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ..., float]]:
                Tuple containing the sample data in various formats depending on the configuration
        """
        user = self.user[idx]  # (batch_size,)
        item = self.item[idx]
        rating = self.rating[idx]

        if self.is_recommender:  # pretraining
            return user, item, rating

        seq = self.seq[idx]  # (batch_size, seq_len)
        feature = self.feature[idx]  # (batch_size, 1)
        feature_neg = self.feature_neg[idx]  # (batch_size, 1)

        pred_rating = 0
        if self.use_aspect:  # erra
            aspect = self.aspect[idx]  # (batch_size, 2x2)
            return user, item, rating, seq, feature, feature_neg, aspect
        if self.use_pred_rating:  # peter
            pred_rating = self.pred_rating[idx]
        # peter, cer
        return user, item, rating, seq, feature, feature_neg, pred_rating


class XRecTokenizerDataset(Dataset):
    """Dataset class for recommendation system data with tokenizer.

    Attributes:
        is_recommender (bool): Whether this is a recommender system
        use_pred_rating (bool): Whether to use predicted ratings
        use_feat_ui_retrieval (bool): Whether to use UI feature retrieval
        user (torch.Tensor): User indices
        item (torch.Tensor): Item indices
        rating (torch.Tensor): Rating values
        feature (torch.Tensor): Feature data
        feature_neg (torch.Tensor): Negative feature data
        seq (torch.Tensor): Sequence data
        mask (torch.Tensor): Attention mask
        fid (torch.Tensor): Feature IDs
        feat_ui (Optional[torch.Tensor]): UI feature data
        mask_feat_ui (Optional[torch.Tensor]): UI feature mask
        pred_rating (Optional[torch.Tensor]): Predicted rating data
    """

    def __init__(
        self,
        data: List[Dict[Any, Any]],
        bos_token: str,
        eos_token: str,
        pad_token: str,
        tokenizer: PreTrainedTokenizer,
        n_features: int = 0,
        is_recommender: bool = False,
    ) -> None:
        """Initialize the XRecTokenizerDataset.

        Args:
            data (List[Dict[Any, Any]]): List of data samples
            bos_token (str): Beginning of sequence token
            eos_token (str): End of sequence token
            pad_token (str): Padding token
            tokenizer (PreTrainedTokenizer): Tokenizer for text processing
            n_features (int, optional): Number of features. Defaults to 0.
            is_recommender (bool, optional): Whether this is a recommender system. Defaults to False.
        """
        self.is_recommender = is_recommender
        self.use_pred_rating = False
        self.use_feat_ui_retrieval = False

        u, i, r, f, a, t, fid, fui, pr = [], [], [], [], [], [], [], [], []

        for x in data:
            u.append(x["user"])
            i.append(x["item"])
            r.append(x["rating"])
            f.append([x["feature"]])
            a.append([x["feature_neg"]])
            t.append(f"{bos_token} {x['text']} {eos_token}")

            idxs = [
                (
                    tokenizer.convert_tokens_to_ids(x["pos_text"])
                    if x["pos_text"] is not None
                    else tokenizer.convert_tokens_to_ids(pad_token)
                ),
                (
                    tokenizer.convert_tokens_to_ids(x["neg_text"])
                    if x["neg_text"] is not None
                    else tokenizer.convert_tokens_to_ids(pad_token)
                ),
            ]
            fid.append(idxs)

            if "retrieved_feats_ui" in x:  # pepler-d
                self.use_feat_ui_retrieval = True
                fui.append(x["retrieved_feats_ui"])

            if "pred_rating" in x:  # pepler, pemdm
                self.use_pred_rating = True
                pr.append(x["pred_rating"])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()

        # NOTE: Words get split with a tokenizer (ex. 'treat' -> ['t', 'reat']) -> word2idx
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.feature_neg = torch.tensor(a, dtype=torch.int64).contiguous()

        encoded_inputs = tokenizer(t, padding=True, return_tensors="pt")
        self.seq = encoded_inputs["input_ids"].contiguous()
        self.mask = encoded_inputs["attention_mask"].contiguous()

        self.fid = torch.tensor(fid, dtype=torch.int64)

        if self.use_feat_ui_retrieval:
            encoded_inputs = tokenizer(fui, padding=True, return_tensors="pt")
            # NOTE: Words get split with a tokenizer -> x [:, :n_features] , o [:, :max_seq_len]
            self.feat_ui = encoded_inputs["input_ids"][
                :, : n_features * 3
            ].contiguous()
            self.mask_feat_ui = encoded_inputs["attention_mask"][
                :, : n_features * 3
            ].contiguous()

        if self.use_pred_rating:
            self.pred_rating = torch.tensor(pr, dtype=torch.float).contiguous()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self.user.size(0)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Union[torch.Tensor, float],
        ],
    ]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ..., float]]:
                Tuple containing the sample data in various formats depending on the configuration
        """
        user = self.user[idx]  # (batch_size,)
        item = self.item[idx]
        rating = self.rating[idx]

        if self.is_recommender:  # pretraining
            return user, item, rating

        feature = self.feature[idx]
        feature_neg = self.feature_neg[idx]
        seq = self.seq[idx]  # (batch_size, seq_len + 2)
        mask = self.mask[idx]

        if self.use_feat_ui_retrieval:  # pepler-d
            feat_ui = self.feat_ui[idx]
            mask_feat_ui = self.mask_feat_ui[idx]
            return (
                feat_ui,
                mask_feat_ui,
                rating,
                seq,
                mask,
                feature,
                feature_neg,
            )
        else:
            pred_rating = 0.0
            if self.use_pred_rating:
                pred_rating = self.pred_rating[idx]
            return (
                user,
                item,
                rating,
                seq,
                mask,
                feature,
                feature_neg,
                pred_rating,
            )  # pepler
