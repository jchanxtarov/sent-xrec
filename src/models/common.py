import copy
import logging
import math
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import lightning as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from metrics.tools import (
    feature_detection,
    get_bert_score,
    get_bleu_score,
    get_feature_coverage_ratio,
    get_feature_diversity,
    get_feature_matching_ratio,
    get_mean_absolute_error,
    get_root_mean_square_error,
    get_rouge_score,
    get_unique_sentence_ratio,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)


class BASE(pl.LightningModule):
    """Base class for all recommendation models.

    This class provides common functionality for training, validation, and testing
    of recommendation models. It includes methods for logging metrics, handling samples,
    and computing various evaluation metrics.

    Attributes:
        storage (ReviewDataLoader): Data loader for reviews
        opt_lr (float): Learning rate
        opt_wd (float): Weight decay
        opt_factor (float): Learning rate reduction factor
        opt_step_size (int): Steps between learning rate updates
        check_gen_text_every_n_epoch (int): Epochs between text generation checks
        check_n_samples (int): Number of samples to check
        save_root (str): Directory to save model outputs
    """

    def __init__(
        self,
        storage: ReviewDataLoader,
        opt_lr: float,
        opt_wd: float,
        opt_factor: float,
        opt_step_size: int,
        check_gen_text_every_n_epoch: int = 1,
        check_n_samples: int = 1,
        save_root: str = "",
    ):
        """Initialize the base model.

        Args:
            storage (ReviewDataLoader): Data loader for reviews
            opt_lr (float): Learning rate
            opt_wd (float): Weight decay
            opt_factor (float): Learning rate reduction factor
            opt_step_size (int): Steps between learning rate updates
            check_gen_text_every_n_epoch (int, optional): Epochs between text generation checks. Defaults to 1
            check_n_samples (int, optional): Number of samples to check. Defaults to 1
            save_root (str, optional): Directory to save model outputs. Defaults to ""
        """
        super().__init__()

        self.storage = storage
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.opt_factor = opt_factor
        self.opt_step_size = opt_step_size
        self.check_gen_text_every_n_epoch = check_gen_text_every_n_epoch
        self.check_n_samples = check_n_samples
        self.save_root = save_root

        self.outputs_test_step: Dict[str, List[Any]] = defaultdict(list)
        self.log_table_samples = []
        self.list_tokens_predict = []

    def log_metrics(self, key: str, val: float, on_step: bool = True) -> None:
        """Log metrics during training.

        Args:
            key (str): Metric name
            val (float): Metric value
            on_step (bool, optional): Whether to log on step. Defaults to True
        """
        self.log(
            key,
            val,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            logger=True,
        )

    def log_samples(
        self,
        rating: List[float],
        rating_pred: Optional[List[float]],
        text: List[str],
        text_pred: List[str],
        pos_negs: Optional[List[str]] = None,
        kws_pos: Optional[List[str]] = None,
        kws_neg: Optional[List[str]] = None,
    ) -> None:
        """Log sample predictions for validation.

        Args:
            rating (List[float]): Ground truth ratings
            rating_pred (Optional[List[float]]): Predicted ratings
            text (List[str]): Ground truth texts
            text_pred (List[str]): Predicted texts
            pos_negs (Optional[List[str]], optional): Positive/negative aspects. Defaults to None
            kws_pos (Optional[List[str]], optional): Positive keywords. Defaults to None
            kws_neg (Optional[List[str]], optional): Negative keywords. Defaults to None
        """
        columns = ["epoch", "rating", "text", "text_pred"]
        samples = [
            [
                self.current_epoch,
                rating[i],
                text[i],
                text_pred[i],
            ]
            for i in range(self.check_n_samples)
        ]

        if rating_pred is not None:
            columns.append("rating_pred")
            samples = [
                sample + [add]
                for sample, add in zip(
                    samples, rating_pred[: self.check_n_samples]
                )
            ]

        if pos_negs is not None:
            columns.append("pos_negs")
            samples = [
                sample + [add]
                for sample, add in zip(
                    samples, pos_negs[: self.check_n_samples]
                )
            ]

        if kws_pos is not None:
            columns.append("kws_pos")
            samples = [
                sample + [add]
                for sample, add in zip(
                    samples, kws_pos[: self.check_n_samples]
                )
            ]

        if kws_neg is not None:
            columns.append("kws_neg")
            samples = [
                sample + [add]
                for sample, add in zip(
                    samples, kws_neg[: self.check_n_samples]
                )
            ]

        if self.logger is not None:
            self.log_table_samples.extend(samples)
            self.logger.log_table(
                key="validation_sample",
                columns=columns,
                data=self.log_table_samples,
            )

    def get_metrics(
        self,
        user: Optional[List[int]],
        item: Optional[List[int]],
        rating: Optional[List[float]],
        rating_predict: Optional[List[float]],
        tokens_test: List[List[str]],
        tokens_predict: List[List[str]],
        text_test: List[str],
        text_predict: List[str],
        feature: torch.Tensor,
        feature_neg: torch.Tensor,
    ) -> Dict[str, float]:
        """Calculate various evaluation metrics.

        Args:
            user (Optional[List[float]]): User indecies
            item (Optional[List[float]]): Item indecies
            rating (Optional[List[float]]): Ground truth ratings
            rating_predict (Optional[List[float]]): Predicted ratings
            tokens_test (List[List[str]]): Ground truth tokens
            tokens_predict (List[List[str]]): Predicted tokens
            text_test (List[str]): Ground truth texts
            text_predict (List[str]): Predicted texts
            feature (torch.Tensor): Feature tensor
            feature_neg (torch.Tensor): Negative feature tensor

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        scores = {}

        if user and item:
            self.outputs_test_step["useridx"] += user
            self.outputs_test_step["itemidx"] += item

        # Rating metrics
        if rating is not None and rating_predict is not None:
            pair_rating = [(r, p) for (r, p) in zip(rating, rating_predict)]
            scores["rmse"] = get_root_mean_square_error(
                pair_rating, self.storage.max_rating, self.storage.min_rating
            )
            scores["mae"] = get_mean_absolute_error(
                pair_rating, self.storage.max_rating, self.storage.min_rating
            )
            self.outputs_test_step["rating"] += rating
            self.outputs_test_step["rating_predict"] += rating_predict

        # Text metrics
        self.list_tokens_predict.extend(tokens_predict)
        scores.update(get_bleu_score(tokens_test, tokens_predict))
        scores.update(get_rouge_score(text_test, text_predict))
        scores.update(get_bert_score(text_test, text_predict))

        # Explainability metrics (positive features)
        feature_pos_test, scores = self.get_explainability_metrics(
            scores,
            feature,
            self.storage.feature_pos_set,
            tokens_predict,
            text_predict,
            "pos",
        )
        # Explainability metrics (negative features)
        feature_neg_test, scores = self.get_explainability_metrics(
            scores,
            feature_neg,
            self.storage.feature_neg_set,
            tokens_predict,
            text_predict,
            "neg",
        )

        # Accumulate results for final logging
        batch_size = len(tokens_test)
        for k, v in scores.items():
            self.outputs_test_step[k].append(v * batch_size)
        self.outputs_test_step["batch_size"].append(batch_size)

        self.outputs_test_step["text"] += text_test
        self.outputs_test_step["text_predict"] += text_predict
        self.outputs_test_step["feature_pos_test"] += feature_pos_test
        self.outputs_test_step["feature_neg_test"] += feature_neg_test

        return scores

    def get_explainability_metrics(
        self,
        scores: Dict[str, float],
        feature: torch.Tensor,
        feature_set: Set[str],
        tokens_predict: List[List[str]],
        text_predict: List[str],
        aspect_type: str = "",
    ) -> Tuple[List[Optional[str]], Dict[str, float]]:
        """Calculate explainability metrics.

        Args:
            scores (Dict[str, float]): Current scores dictionary
            feature (torch.Tensor): Feature tensor
            feature_set (Set[str]): Set of features
            tokens_predict (List[List[str]]): Predicted tokens
            text_predict (List[str]): Predicted texts
            aspect_type (str, optional): Type of features. Defaults to ""

        Returns:
            Tuple[List[Optional[str]], Dict[str, float]]:
                - Feature test results as a list of feature tokens or None
                - Updated scores dictionary
        """
        aspect_type = f"_{aspect_type}" if aspect_type else ""

        feature_batch = feature_detection(tokens_predict, feature_set)
        feature_test = self.feature_id2token(feature)

        # Feature Matching Ratio
        scores[f"fmr{aspect_type}"] = get_feature_matching_ratio(
            feature_batch, feature_test
        )

        return feature_test, scores

    def get_explainability_metrics_on_test_end(
        self,
        scores: Dict[Any, Any],
        feature_set: Set[str],
        tokens_predict: List[List[str]],
        aspect_type: str = "",
    ) -> Dict[Any, Any]:
        """Accumulate final explainability metrics on test end.

        Args:
            scores (Dict[Any, Any]): Current scores dictionary
            feature_set (Set[str]): Set of features (pos or neg)
            tokens_predict (List[List[str]]): Predicted tokens
            aspect_type (str, optional): Type of features. Defaults to ""

        Returns:
            Dict[Any, Any]: Updated scores dictionary
        """
        aspect_type = f"_{aspect_type}" if aspect_type else ""
        feature_batch = feature_detection(tokens_predict, feature_set)

        # Diversity and coverage
        scores[f"div{aspect_type}"] = get_feature_diversity(feature_batch)
        scores[f"fcr{aspect_type}"] = get_feature_coverage_ratio(
            feature_batch, feature_set
        )
        # Unique Sentence Ratio
        scores["usr"], scores["usn"] = get_unique_sentence_ratio(
            tokens_predict
        )
        return scores

    def feature_id2token(self, feature: torch.Tensor) -> List[Optional[str]]:
        """Convert feature IDs to tokens.

        Args:
            feature (torch.Tensor): Feature tensor of IDs

        Returns:
            List[Optional[str]]: Feature tokens or None for padding
        """
        feature_test = [
            self.storage.word_dict.idx2word[i] if i != -100 else None
            for i in feature.view(-1).tolist()
        ]  # ids to words
        return feature_test

    def on_test_epoch_end(self) -> Any:
        """Compute final test metrics and optionally save results.

        Returns:
            Any: Scores dictionary
        """
        # Remove unused columns that never get populated
        non_score_cols = [
            "rating",
            "rating_predict",
            "rating_input",
            "text",
            "text_predict",
            "feature_pos_test",
            "feature_neg_test",
            "text_prof_u",
            "text_prof_i",
            "text_kw_ui_pos",
            "text_kw_ui_neg",
        ]

        total = sum(self.outputs_test_step["batch_size"])
        scores = {
            k: sum(v) / total
            for k, v in self.outputs_test_step.items()
            if k not in non_score_cols and k != "batch_size"
        }

        # Finalize explainability metrics
        scores = self.get_explainability_metrics_on_test_end(
            scores,
            self.storage.feature_pos_set,
            self.list_tokens_predict,
            "pos",
        )
        scores = self.get_explainability_metrics_on_test_end(
            scores,
            self.storage.feature_neg_set,
            self.list_tokens_predict,
            "neg",
        )

        # Log final scores
        for k, v in scores.items():
            self.log(k, v)

        # Save partial results if requested
        results = {
            k: v
            for k, v in self.outputs_test_step.items()
            if k in non_score_cols
        }
        if self.save_root:
            pd.DataFrame(results).to_csv(
                f"{self.save_root}/{self.save_root.rsplit('/', 1)[-1]}_result.csv",
                index=False,
            )

        return scores


def ids2tokens(
    idxs: List[int], word2idx: Dict[str, int], idx2word: Dict[int, str]
) -> List[str]:
    """Convert token indices to tokens using word dictionaries.

    Args:
        idxs (List[int]): List of token indices
        word2idx (Dict[str, int]): Word to index mapping
        idx2word (Dict[int, str]): Index to word mapping

    Returns:
        List[str]: List of tokens
    """
    tokens = []
    for idx in idxs:
        if idx == word2idx["<eos>"]:
            break
        tokens.append(idx2word[idx])
    return tokens


def ids2tokens_tokenizer(
    idxs: List[int], tokenizer: PreTrainedTokenizer
) -> List[str]:
    """Convert token indices to tokens using a pre-trained tokenizer.

    Args:
        idxs (List[int]): List of token indices
        tokenizer (PreTrainedTokenizer): Pre-trained tokenizer

    Returns:
        List[str]: List of decoded tokens
    """
    text = __postprocessing(tokenizer.decode(idxs))
    if text == "":  # for null feature
        return [""]
    tokens = []
    for token in text.split():
        if "<eos>" in token or "<pad>" in token:
            break
        tokens.append(token)
    return tokens


def __postprocessing(string: str) -> str:
    """Post-process decoded text by standardizing spacing and punctuation.

    Args:
        string (str): Input string to process

    Returns:
        str: Processed string
    """
    # see also: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    patterns = [
        ("'s", " 's"),
        ("'m", " 'm"),
        ("'ve", " 've"),
        ("n't", " n't"),
        ("'re", " 're"),
        ("'d", " 'd"),
        ("'ll", " 'll"),
        (r"\(", " ( "),
        (r"\)", " ) "),
        (",+", " , "),
        (":+", " , "),
        (";+", " . "),
        (r"\.+", " . "),
        ("!+", " ! "),
        (r"\?+", " ? "),
        (" +", " "),
    ]
    for pattern, replacement in patterns:
        string = re.sub(pattern, replacement, string)
    return string.strip()


def get_square_subsequent_mask(total_len: int) -> torch.Tensor:
    """Create a square subsequent mask for transformer attention.

    Args:
        total_len (int): Length of the sequence

    Returns:
        torch.Tensor: Attention mask tensor
    """
    mask = torch.tril(
        torch.ones(total_len, total_len)
    )  # (total_len, total_len), lower triangle -> 1.; others -> 0
    mask = mask == 0  # lower -> False; others -> True
    return mask


def get_peter_mask(src_len: int, tgt_len: int) -> torch.Tensor:
    """Create an attention mask for the PETER model.

    Args:
        src_len (int): Source sequence length
        tgt_len (int): Target sequence length

    Returns:
        torch.Tensor: PETER attention mask tensor
    """
    total_len = src_len + tgt_len
    mask = get_square_subsequent_mask(total_len)
    # allow user to attend item
    mask[0, 1] = False
    return mask


def get_erra_mask(src_len: int, tgt_len: int) -> torch.Tensor:
    """Create an attention mask for the ERRA model.

    Args:
        src_len (int): Source sequence length
        tgt_len (int): Target sequence length

    Returns:
        torch.Tensor: ERRA attention mask tensor
    """
    total_len = src_len + tgt_len
    mask = get_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[0, 3] = False
    mask[1, 2] = False
    mask[1, 3] = False
    mask[2, 3] = False
    return mask


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer implementation.

    This class implements a single transformer encoder layer with self-attention
    and feed-forward networks.

    Attributes:
        self_attn (nn.MultiheadAttention): Multi-head self-attention layer
        linear1 (nn.Linear): First linear layer of feed-forward network
        dropout (nn.Dropout): Dropout layer
        linear2 (nn.Linear): Second linear layer of feed-forward network
        norm1 (nn.LayerNorm): Layer normalization for attention output
        norm2 (nn.LayerNorm): Layer normalization for feed-forward output
        dropout1 (nn.Dropout): Dropout after attention
        dropout2 (nn.Dropout): Dropout after feed-forward
        activation (callable): Activation function
    """

    def __init__(
        self,
        d_embed: int,
        n_head: int,
        d_feedforward: int = 2048,
        dropout_ratio: float = 0.1,
        activation_type: str = "relu",
    ):
        """Initialize the transformer encoder layer.

        Args:
            d_embed (int): Embedding dimension
            n_head (int): Number of attention heads
            d_feedforward (int, optional): Feed-forward network dimension. Defaults to 2048
            dropout_ratio (float, optional): Dropout ratio. Defaults to 0.1
            activation_type (str, optional): Activation function type. Defaults to "relu"
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_embed, n_head, dropout=dropout_ratio
        )
        self.linear1 = nn.Linear(d_embed, d_feedforward)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(d_feedforward, d_embed)
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.activation = get_activation_fn(activation_type)

    def __setstate__(self, state: Dict[Any, Any]):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer encoder layer.

        Args:
            src (torch.Tensor): Source sequence
            src_mask (Optional[torch.Tensor], optional): Source attention mask
            src_key_padding_mask (Optional[torch.Tensor], optional): Source key padding mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        src2, attn = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.

    This class implements a stack of transformer encoder layers with optional normalization.

    Attributes:
        layers (nn.ModuleList): List of transformer encoder layers
        n_layers (int): Number of layers
        norm (Optional[Any]): Optional normalization layer
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: nn.Module,
        n_layers: int,
        norm: Optional[Any] = None,
    ):
        """Initialize the transformer encoder.

        Args:
            encoder_layer (nn.Module): Transformer encoder layer
            n_layers (int): Number of layers
            norm (Optional[Any], optional): Normalization layer
        """
        super().__init__()
        self.layers = get_clones(encoder_layer, n_layers)
        self.n_layers = n_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer encoder.

        Args:
            src (torch.Tensor): Source sequence
            mask (Optional[torch.Tensor], optional): Attention mask
            src_key_padding_mask (Optional[torch.Tensor], optional): Key padding mask

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


def get_clones(module: nn.Module, n_layers: int):
    """Clone a module `n_layers` times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def get_activation_fn(activation_type: str) -> Callable:
    """Get activation function by name.

    Args:
        activation_type (str): Name of the activation function

    Returns:
        Callable: The requested activation function

    Raises:
        RuntimeError: If the activation type is not supported
    """
    if activation_type == "relu":
        return F.relu
    elif activation_type == "gelu":
        return F.gelu
    else:
        raise RuntimeError(
            f"activation should be relu/gelu, not {activation_type}"
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(
        self, d_embed: int, dropout_ratio: float = 0.1, max_len: int = 5000
    ):
        """Initialize the positional encoding.

        Args:
            d_embed (int): Embedding dimension
            dropout_ratio (float, optional): Dropout ratio. Defaults to 0.1
            max_len (int, optional): Maximum sequence length. Defaults to 5000
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)

        p_emb = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )  # (d_embed/2,)

        p_emb[:, 0::2] = torch.sin(
            position * div_term
        )  # even number index, (max_len, d_embed/2)
        p_emb[:, 1::2] = torch.cos(position * div_term)  # odd number index
        p_emb = p_emb.unsqueeze(0).transpose(
            0, 1
        )  # (max_len, d_embed) -> (1, max_len, d_embed) -> (max_len, 1, d_embed)
        self.register_buffer(
            "p_emb", p_emb
        )  # NOTE: Not updated by back-propagation; accessible via its name.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Tensor with positional encoding added
        """
        x = x + self.p_emb[: x.size(0), :]
        return self.dropout(x)


class MLPRating(nn.Module):
    """Multi-layer perceptron for rating prediction.

    This class implements a flexible MLP for predicting ratings.

    Attributes:
        first_layer (nn.Linear): First linear layer
        hidden_layers (Optional[nn.ModuleList]): Optional hidden layers
        last_layer (nn.Linear): Output layer
        sigmoid (nn.Sigmoid): Sigmoid activation
    """

    def __init__(
        self,
        d_embed: int = 512,
        width: int = 1,
        n_hidden_layers: int = 0,
        d_hidden: int = 400,
    ):
        """Initialize the MLP rating predictor.

        Args:
            d_embed (int, optional): Embedding dimension. Defaults to 512
            width (int, optional): Input width multiplier. Defaults to 1
            n_hidden_layers (int, optional): Number of hidden layers. Defaults to 0
            d_hidden (int, optional): Hidden layer dimension. Defaults to 400
        """
        super().__init__()
        if n_hidden_layers > 0:
            self.first_layer = nn.Linear(d_embed * width, d_hidden)
            layer = nn.Linear(d_hidden, d_hidden)
            self.hidden_layers = get_clones(layer, n_hidden_layers)
            self.last_layer = nn.Linear(d_hidden, 1)
        else:
            self.first_layer = nn.Linear(d_embed * width, d_embed)
            self.hidden_layers = None
            self.last_layer = nn.Linear(d_embed, 1)

        self.sigmoid = nn.Sigmoid()
        self.__init_weights()

    def __init_weights(self):
        init_range = 0.1
        self.first_layer.weight.data.uniform_(-init_range, init_range)
        self.first_layer.bias.data.zero_()
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                layer.weight.data.uniform_(-init_range, init_range)
                layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-init_range, init_range)
        self.last_layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input features

        Returns:
            torch.Tensor: Predicted rating
        """
        x = self.sigmoid(self.first_layer(x))  # (batch_size, d_embed)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                x = self.sigmoid(layer(x))  # (batch_size, d_hidden)
        return torch.squeeze(self.last_layer(x))  # (batch_size,)


class MFRating(nn.Module):
    """Matrix Factorization based rating predictor."""

    def __init__(self):
        """Initialize the MF rating predictor."""
        super().__init__()

    def forward(
        self, u_src: torch.Tensor, i_src: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the MF rating predictor.

        Args:
            u_src (torch.Tensor): User embeddings
            i_src (torch.Tensor): Item embeddings

        Returns:
            torch.Tensor: Predicted ratings
        """
        # TODO: standardization?, 5 -> max_rating
        rating = 5 * torch.clamp(
            torch.sum(u_src * i_src, 1), min=0
        )  # (batch_size,)
        return rating


class TransformerRating(nn.Module):
    """Transformer-based rating predictor.

    This class implements a transformer model for rating prediction by processing
    user and item embeddings through transformer layers.
    """

    def __init__(
        self,
        d_embed: int = 512,
        n_head: int = 4,
        d_hidden: int = 400,
        n_layers: int = 2,
    ):
        """Initialize the transformer rating predictor.

        Args:
            d_embed (int, optional): Embedding dimension. Defaults to 512
            n_head (int, optional): Number of attention heads. Defaults to 4
            d_hidden (int, optional): Hidden layer dimension. Defaults to 400
            n_layers (int, optional): Number of transformer layers. Defaults to 2
        """
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_embed=d_embed,
            n_head=n_head,
            d_feedforward=d_hidden,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers=n_layers
        )
        self.fc = nn.Linear(d_embed, 1)

    def forward(
        self, user_embed: torch.Tensor, item_embed: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the transformer rating predictor.

        Args:
            user_embed (torch.Tensor): User embeddings
            item_embed (torch.Tensor): Item embeddings

        Returns:
            torch.Tensor: Predicted ratings
        """
        x = torch.cat(
            (user_embed.unsqueeze(0), item_embed.unsqueeze(0)), dim=0
        )
        x = self.transformer_encoder(x)  # (2, batch_size, d_embed)
        if isinstance(x, tuple):
            x = x[0]
        x = torch.mean(x, dim=0)  # (batch_size, d_embed)
        rating = self.fc(x).squeeze(-1)  # (batch_size)
        return rating


class Recommender(BASE):
    """A recommendation model that combines user and item embeddings for rating prediction.

    This class implements a recommendation model that can use different architectures (MF, MLP, Transformer)
    for predicting ratings based on user and item embeddings.

    Attributes:
        user_embeddings (nn.Embedding): Embedding layer for users
        item_embeddings (nn.Embedding): Embedding layer for items
        recommender (Union[MFRating, MLPRating, TransformerRating]): The rating prediction model
        rec_type (str): Type of recommender model ('mf', 'mlp', or 'transformer')
        opt_wd (float): Weight decay for optimization
        criterion_rating (nn.MSELoss): Loss function for rating prediction
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        storage: ReviewDataLoader,
        d_embed: Optional[int] = 512,
        pretrained_embed_name: Optional[str] = "gpt2",
        rec_type: Optional[str] = "mlp",
        n_hidden_layers: int = 2,
        d_hidden: int = 400,
        opt_lr: float = 0.1,
        opt_wd: float = 1e-4,
        opt_factor: float = 0.25,
        opt_step_size: int = 1,
        save_root: str = "",
    ) -> None:
        """Initialize the recommender model.

        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset
            storage (ReviewDataLoader): Data loader for reviews
            d_embed (Optional[int], optional): Embedding dimension. Defaults to 512
            pretrained_embed_name (Optional[str], optional): Name of pretrained model. Defaults to "gpt2"
            rec_type (Optional[str], optional): Type of recommender model. Defaults to "mlp"
            n_hidden_layers (int, optional): Number of hidden layers. Defaults to 2
            d_hidden (int, optional): Hidden layer dimension. Defaults to 400
            opt_lr (float, optional): Learning rate. Defaults to 0.1
            opt_wd (float, optional): Weight decay. Defaults to 1e-4
            opt_factor (float, optional): Learning rate reduction factor. Defaults to 0.25
            opt_step_size (int, optional): Steps between learning rate updates. Defaults to 1
            save_root (str, optional): Directory to save outputs. Defaults to ""
        """
        super().__init__(
            storage,
            opt_lr,
            opt_wd,
            opt_factor,
            opt_step_size,
            0,
            0,
            save_root,
        )

        assert (
            d_embed is not None or pretrained_embed_name is not None
        ), "cannot set dim embed."

        # If no embedding dimension is provided, infer from GPT2
        if d_embed is None:
            lm = GPT2LMHeadModel.from_pretrained(pretrained_embed_name)
            d_embed = lm.transformer.wte.weight.size(1)

        self.user_embeddings = nn.Embedding(n_users, d_embed)
        self.item_embeddings = nn.Embedding(n_items, d_embed)
        self.__initialize_tokens()

        self.rec_type = rec_type
        if rec_type == "mf":
            self.recommender = MFRating()
        elif rec_type == "mlp":
            self.recommender = MLPRating(
                d_embed=d_embed,
                width=2,
                n_hidden_layers=n_hidden_layers,
                d_hidden=d_hidden,
            )
        elif rec_type == "transformer":
            self.recommender = TransformerRating(
                d_embed=d_embed,
                n_head=4,
                d_hidden=d_hidden,
                n_layers=n_hidden_layers,
            )

        self.opt_wd = opt_wd
        self.criterion_rating = nn.MSELoss()

    def __initialize_tokens(self) -> None:
        """Initialize the user and item embedding weights."""
        init_range = 0.1
        self.user_embeddings.weight.data.uniform_(-init_range, init_range)
        self.item_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Forward pass of the recommender model.

        Args:
            user (torch.Tensor): User indices
            item (torch.Tensor): Item indices

        Returns:
            torch.Tensor: Predicted ratings
        """
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)

        if self.rec_type == "mf":
            rating = self.recommender(u_src, i_src)
        elif self.rec_type == "mlp":
            ui_src = torch.cat([u_src, i_src], 1)
            rating = self.recommender(ui_src)
        else:  # self.rec_type == "transformer"
            rating = self.recommender(u_src, i_src)

        return rating

    def lossfun(
        self, rating_pred: torch.Tensor, rating_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate the loss between predicted and ground truth ratings.

        Args:
            rating_pred (torch.Tensor): Predicted ratings
            rating_gt (torch.Tensor): Ground truth ratings

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value
        """
        loss = self.criterion_rating(rating_pred, rating_gt)
        return {"loss": loss}

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Batch of (user, item, rating)
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss value for the batch
        """
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        output = self.lossfun(rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"pretrain/train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Batch of (user, item, rating)
            batch_idx (int): Index of the batch
        """
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        output = self.lossfun(rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"pretrain/valid/{k}", v, on_step=False)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Batch of (user, item, rating)
            batch_idx (int): Index of the batch
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0
        """
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        pair_rating = [
            (r, p.item()) for (r, p) in zip(rating.tolist(), rating_pred)
        ]

        # Store metrics
        self.outputs_test_step["pretrain/rmse"].append(
            get_root_mean_square_error(
                pair_rating, self.storage.max_rating, self.storage.min_rating
            )
        )
        self.outputs_test_step["pretrain/mae"].append(
            get_mean_absolute_error(
                pair_rating, self.storage.max_rating, self.storage.min_rating
            )
        )
        self.outputs_test_step["rating"] += rating.tolist()
        self.outputs_test_step["rating_predict"] += [
            r.item() for r in rating_pred
        ]

    def on_test_epoch_end(self) -> None:
        """Process and log metrics at the end of the test epoch."""
        non_score_cols = ["rating", "rating_predict"]
        scores = {
            k: mean(v)
            for k, v in self.outputs_test_step.items()
            if k not in non_score_cols
        }
        for k, v in scores.items():
            self.log(k, v)

        results = {
            k: v
            for k, v in self.outputs_test_step.items()
            if k in non_score_cols
        }
        if self.save_root:
            pd.DataFrame(results).to_csv(
                f"{self.save_root}/result_pretrain.csv", index=False
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer for training.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        return {
            "optimizer": optimizer,
            "monitor": "pretrain/valid/loss",
        }
