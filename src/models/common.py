import copy
import math
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

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


class BASE(pl.LightningModule):

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

    def log_metrics(self, key: int, val: float, on_step: bool = True):
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
        rating,
        rating_pred,
        text,
        text_pred,
        pos_negs=None,
        kws_pos=None,
        kws_neg=None,
    ):
        colmuns = ["epoch", "rating", "text", "text_pred"]
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
            colmuns.append("rating_pred")
            samples = [
                sample + [add]
                for sample, add in zip(samples, rating_pred[: self.check_n_samples])
            ]

        if pos_negs is not None:
            colmuns.append("pos_negs")
            samples = [
                sample + [add]
                for sample, add in zip(samples, pos_negs[: self.check_n_samples])
            ]

        if kws_pos is not None:
            colmuns.append("kws_pos")
            samples = [
                sample + [add]
                for sample, add in zip(samples, kws_pos[: self.check_n_samples])
            ]

        if kws_neg is not None:
            colmuns.append("kws_neg")
            samples = [
                sample + [add]
                for sample, add in zip(samples, kws_neg[: self.check_n_samples])
            ]

        if self.logger is not None:
            self.log_table_samples.extend(samples)
            self.logger.log_table(
                key="validation_sample",
                columns=colmuns,
                data=self.log_table_samples,
            )

    def get_metrics(
        self,
        rating: Optional[List[float]],
        rating_predict: Optional[List[float]],
        tokens_test: List[List[str]],
        tokens_predict: List[List[str]],
        text_test: List[str],
        text_predict: List[str],
        feature: torch.Tensor,
        feature_neg: torch.Tensor,
    ) -> dict:
        scores = {}

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

        self.list_tokens_predict.extend(tokens_predict)
        scores.update(get_bleu_score(tokens_test, tokens_predict))
        scores.update(get_rouge_score(text_test, text_predict))
        scores.update(get_bert_score(text_test, text_predict))

        feature_pos_test, scores = self.get_explainability_metrics(
            scores,
            feature,
            self.storage.feature_pos_set,
            tokens_predict,
            text_predict,
            "pos",
        )
        feature_neg_test, scores = self.get_explainability_metrics(
            scores,
            feature_neg,
            self.storage.feature_neg_set,
            tokens_predict,
            text_predict,
            "neg",
        )
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
        type: str = "",
    ) -> Tuple[List[Optional[str]], dict]:
        type = f"_{type}" if type != "" else ""

        feature_batch = feature_detection(tokens_predict, feature_set)
        feature_test = self.feature_id2token(feature)
        scores[f"fmr{type}"] = get_feature_matching_ratio(feature_batch, feature_test)

        return feature_test, scores

    def get_explainability_metrics_on_test_end(
        self,
        scores: Dict[Any, Any],
        feature_set: Set[str],
        tokens_predict: List[List[str]],
        type: str = "",
    ) -> Dict[Any, Any]:
        type = f"_{type}" if type != "" else ""
        feature_batch = feature_detection(tokens_predict, feature_set)
        scores[f"div{type}"] = get_feature_diversity(feature_batch)
        scores[f"fcr{type}"] = get_feature_coverage_ratio(feature_batch, feature_set)
        scores["usr"], scores["usn"] = get_unique_sentence_ratio(tokens_predict)
        return scores

    def feature_id2token(self, feature: torch.Tensor) -> List[Optional[str]]:
        feature_test = [
            self.storage.word_dict.idx2word[i] if i != -100 else None
            for i in feature.view(-1).tolist()
        ]  # ids to words
        return feature_test

    def on_test_epoch_end(self) -> Any:
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
        for k, v in scores.items():
            self.log(k, v)

        results = {
            k: v for k, v in self.outputs_test_step.items() if k in non_score_cols
        }
        if self.save_root != "":
            pd.DataFrame(results).to_csv(
                f"{self.save_root}/{self.save_root.rsplit('/', 1)[-1]}_result.csv",
                index=False,
            )

        return scores


def ids2tokens(
    idxs: List[int], word2idx: Dict[str, int], idx2word: Dict[int, str]
) -> List[str]:
    tokens = []
    for idx in idxs:
        if idx == word2idx["<eos>"]:
            break
        tokens.append(idx2word[idx])
    return tokens


def ids2tokens_tokenizer(idxs: List[int], tokenizer: PreTrainedTokenizer) -> List[str]:
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
    string = string.strip()
    return string


def get_square_subsequent_mask(total_len: int) -> torch.Tensor:
    mask = torch.tril(
        torch.ones(total_len, total_len)
    )  # (total_len, total_len), lower triangle -> 1.; others -> 0
    mask = mask == 0  # lower -> False; others -> True
    return mask


def get_peter_mask(src_len: int, tgt_len: int) -> torch.Tensor:
    total_len = src_len + tgt_len
    mask = get_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    return mask


def get_erra_mask(src_len: int, tgt_len: int) -> torch.Tensor:
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

    def __init__(
        self,
        d_embed: int,
        n_head: int,
        d_feedforward: int = 2048,
        dropout_ratio: float = 0.1,
        activation_type: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_embed, n_head, dropout=dropout_ratio)
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

        src2, attn = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):

    __constants__ = ["norm"]

    def __init__(
        self, encoder_layer: nn.Module, n_layers: int, norm: Optional[Any] = None
    ):
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

        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


def get_clones(module: nn.Module, n_layers: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def get_activation_fn(activation_type: str):
    if activation_type == "relu":
        return F.relu
    elif activation_type == "gelu":
        return F.gelu

    raise RuntimeError(
        f"activation_type: {activation_type} is not included in the default options."
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed: int, dropout_ratio: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)

        p_emb = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed)
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

    def forward(self, x: torch.Tensor):
        x = x + self.p_emb[: x.size(0), :]
        return self.dropout(x)


class MLPRating(nn.Module):
    def __init__(
        self,
        d_embed: int = 512,
        width: int = 1,
        n_hidden_layers: int = 0,
        d_hidden: int = 400,
    ):
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

    def forward(self, x: torch.Tensor):  # (batch_size, d_embed)
        x = self.sigmoid(self.first_layer(x))  # (batch_size, d_embed)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                x = self.sigmoid(layer(x))  # (batch_size, d_hidden)
        rating = torch.squeeze(self.last_layer(x))  # (batch_size,)
        return rating


class MFRating(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u_src: torch.Tensor, i_src: torch.Tensor):  # (batch_size, emsize)
        # TODO: standalization?, 5 -> max_rating
        rating = 5 * torch.clamp(torch.sum(u_src * i_src, 1), min=0)  # (batch_size,)
        return rating


class TransformerRating(nn.Module):
    def __init__(
        self,
        d_embed: int = 512,
        n_head: int = 4,
        d_hidden: int = 400,
        n_layers: int = 2,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_embed=d_embed,
            n_head=n_head,
            d_feedforward=d_hidden,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers=n_layers)
        self.fc = nn.Linear(d_embed, 1)

    def forward(self, user_embed: torch.Tensor, item_embed: torch.Tensor):
        x = torch.cat((user_embed.unsqueeze(0), item_embed.unsqueeze(0)), dim=0)
        x = self.transformer_encoder(x)  # (2, batch_size, d_embed)
        if isinstance(x, tuple):
            x = x[0]
        x = torch.mean(x, dim=0)  # (batch_size, d_embed)
        rating = self.fc(x).squeeze(-1)  # (batch_size)
        return rating


class Recommender(BASE):

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
    ):
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

    def __initialize_tokens(self):
        init_range = 0.1
        self.user_embeddings.weight.data.uniform_(-init_range, init_range)
        self.item_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, user, item) -> List[float]:
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        rating = None
        if self.rec_type == "mf":
            rating = self.recommender(u_src, i_src)
        elif self.rec_type == "mlp":
            ui_src = torch.cat([u_src, i_src], 1)
            rating = self.recommender(ui_src)
        elif self.rec_type == "transformer":
            rating = self.recommender(u_src, i_src)
        assert rating is not None, "rating should be included"
        return rating

    def lossfun(self, rating_pred, rating_gt):
        loss = self.criterion_rating(rating_pred, rating_gt)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        output = self.lossfun(rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"pretrain/train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(self, batch, batch_idx):
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        output = self.lossfun(rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"pretrain/valid/{k}", v, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        user, item, rating = batch
        rating_pred = self.forward(user, item)
        pair_rating = [(r, p.item()) for (r, p) in zip(rating.tolist(), rating_pred)]

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
        self.outputs_test_step["rating_predict"] += [r.item() for r in rating_pred]

    def on_test_epoch_end(self):
        non_score_cols = ["rating", "rating_predict"]

        scores = {
            k: mean(v)
            for k, v in self.outputs_test_step.items()
            if k not in non_score_cols
        }
        for k, v in scores.items():
            self.log(k, v)

        results = {
            k: v for k, v in self.outputs_test_step.items() if k in non_score_cols
        }
        if self.save_root != "":
            pd.DataFrame(results).to_csv(
                f"{self.save_root}/result_pretrain.csv", index=False
            )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        return {
            "optimizer": optimizer,
            "monitor": "pretrain/valid/loss",
        }
