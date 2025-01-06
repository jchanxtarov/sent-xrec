import math
from statistics import mean
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders.helpers import ReviewDataLoader
from models.common import (
    BASE,
    MLPRating,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
    get_peter_mask,
    get_square_subsequent_mask,
    ids2tokens,
)


# NOTE: use_feature: (false, true) -> (PETER, PETER+)
class PETER(BASE):

    def __init__(
        self,
        d_embed: int,
        n_head: int,
        n_hid: int,
        n_layers: int,
        peter_mask: bool,
        n_users: int,
        n_items: int,
        src_len: int,
        n_tokens: int,
        pad_idx: int,
        storage: ReviewDataLoader,
        type_rating_embedding: Optional[int] = None,
        use_feature: bool = False,
        max_seq_len: int = 15,
        min_rating: int = 1,
        max_rating: int = 5,
        reg_text: float = 1.0,
        reg_context: float = 1.0,
        reg_rating: float = 0.1,
        dropout: float = 0.2,
        opt_lr: float = 0.1,
        opt_wd: float = 0.0,
        opt_factor: float = 0.25,
        opt_step_size: int = 1,
        check_gen_text_every_n_epoch: int = 10,
        check_n_samples: int = 3,
        save_root: str = "",
    ):
        super().__init__(
            storage,
            opt_lr,
            opt_wd,
            opt_factor,
            opt_step_size,
            check_gen_text_every_n_epoch,
            check_n_samples,
            save_root,
        )

        self.uir_len = 3 if type_rating_embedding is not None else 2
        self.d_embed = d_embed
        self.src_len = src_len
        self.tgt_len = max_seq_len + 1  # added <eos>
        self.n_tokens = n_tokens
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.pad_idx = pad_idx
        self.type_rating_embedding = type_rating_embedding
        self.use_feature = use_feature
        self.max_seq_len = max_seq_len
        self.reg_text = reg_text
        self.reg_context = reg_context
        self.reg_rating = reg_rating
        self.pos_encoder = PositionalEncoding(d_embed, dropout)
        encoder_layers = TransformerEncoderLayer(d_embed, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.user_embeddings = nn.Embedding(n_users, d_embed)
        self.item_embeddings = nn.Embedding(n_items, d_embed)

        if self.type_rating_embedding == 0:
            self.rating_embedding = nn.Embedding(1, d_embed)
        elif self.type_rating_embedding == 1:
            self.n_rating_embedding = self.max_rating
            self.rating_embedding = nn.Embedding(self.n_rating_embedding + 1, d_embed)

        self.word_embeddings = nn.Embedding(n_tokens, d_embed)
        self.hidden2token = nn.Linear(d_embed, n_tokens)
        self.base_recommender = MLPRating(d_embed)

        if peter_mask:
            self.attn_mask = get_peter_mask(self.src_len, self.tgt_len)
        else:
            self.attn_mask = get_square_subsequent_mask(self.src_len + self.tgt_len)

        self.initialize_tokens()

        self.criterion_text = nn.NLLLoss(
            ignore_index=pad_idx
        )  # ignore the padding when computing loss
        self.criterion_rating = nn.MSELoss()

    def initialize_tokens(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

        if self.type_rating_embedding is not None:
            nn.init.xavier_uniform_(self.rating_embedding.weight)

    def forward(
        self,
        user,
        item,
        pre_pred_rating,
        text,
        seq_prediction=True,
        context_prediction=True,
        rating_prediction=True,
    ):
        device = user.device
        batch_size = user.size(0)
        total_len = self.uir_len + text.size(
            1
        )  # deal with generation when total_len != src_len + tgt_len
        attn_mask = self.attn_mask[:total_len, :total_len].to(
            device
        )  # (total_len, total_len)
        left = (
            torch.zeros(batch_size, self.uir_len).bool().to(device)
        )  # (batch_size, uir_len)
        right = text == self.pad_idx
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, d_embed)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, d_embed)
        src = torch.cat([u_src, i_src], 0)

        if self.type_rating_embedding is not None:
            if self.type_rating_embedding == 0:
                r_src = self.rating_embedding.weight.data.unsqueeze(0).expand(
                    -1, batch_size, -1
                ) * pre_pred_rating.unsqueeze(0).unsqueeze(2)
            elif self.type_rating_embedding == 1:
                pre_pred_rating = torch.clamp(
                    pre_pred_rating, min=self.min_rating, max=self.max_rating
                )
                pre_pred_rating = torch.round(
                    (pre_pred_rating / int(self.max_rating / self.n_rating_embedding))
                ).to(torch.int)
                r_src = self.rating_embedding(pre_pred_rating).unsqueeze(0)
            src = torch.cat([src, r_src], 0)

        w_src = torch.transpose(
            self.word_embeddings(text), 0, 1
        )  # (total_len - uir_len, batch_size, d_embed)
        src = torch.cat([src, w_src], 0)  # (total_len, batch_size, d_embed)
        src = src * math.sqrt(self.d_embed)
        src = self.pos_encoder(src)

        hidden, attns = self.transformer_encoder(
            src, attn_mask, key_padding_mask
        )  # (total_len, batch_size, d_embed), (n_layers, batch_size, total_len_tgt, total_len_src)

        rating, log_context_dis = None, None
        if rating_prediction:
            rating = self.predict_rating(hidden)  # (batch_size,)
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)

        if seq_prediction:
            word_prob, log_word_prob = self.predict_sequence(
                hidden
            )  # (tgt_len, batch_size, ntoken)
        else:
            word_prob, log_word_prob = self.generate_token(
                hidden
            )  # (batch_size, ntoken)

        return word_prob, log_word_prob, log_context_dis, rating, attns

    def lossfun(
        self,
        log_context_dis,
        word_prob,
        log_word_prob,
        seq,
        rating_pred,
        rating,
        asp=None,
    ):
        context_dis = log_context_dis.unsqueeze(0).repeat(
            (self.tgt_len - 1, 1, 1)
        )  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        loss_t = self.criterion_text(
            log_word_prob.view(-1, self.n_tokens),
            seq[:, 1:].permute(1, 0).reshape((-1,)),
        )
        loss_c = self.criterion_text(
            context_dis.view(-1, self.n_tokens),
            seq[:, 1:-1].permute(1, 0).reshape((-1,)),
        )
        loss_r = self.criterion_rating(rating_pred, rating)
        loss = (
            self.reg_text * loss_t
            + self.reg_context * loss_c
            + self.reg_rating * loss_r
        )
        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_c": loss_c,
            "loss_r": loss_r,
        }

    def training_step(self, batch, batch_idx):
        user, item, rating, seq, feature, _, pre_pred_rating = batch

        # (test)
        if batch_idx % 1000 == 0:
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, pre_pred_rating, seq, feature
            )
            print(
                f"[test] (train) batch_idx: {batch_idx} | text: {text[0]} | text_pred: {text_pred[0]}"
            )
        if self.use_feature:
            text = torch.cat(
                [feature, seq[:, :-1]], 1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        word_prob, log_word_prob, log_context_dis, rating_pred, _ = self.forward(
            user, item, pre_pred_rating, text
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis, word_prob, log_word_prob, seq, rating_pred, rating, feature
        )

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(self, batch, batch_idx):
        user, item, rating, seq, feature, _, pre_pred_rating = batch

        if (
            self.current_epoch % self.check_gen_text_every_n_epoch == 0
            and batch_idx == 0
        ):
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, pre_pred_rating, seq, feature
            )
            self.log_samples(rating.tolist(), rating_pred, text, text_pred)

        if self.use_feature:
            text = torch.cat(
                [feature, seq[:, :-1]], 1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        word_prob, log_word_prob, log_context_dis, rating_pred, _ = self.forward(
            user, item, pre_pred_rating, text
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis, word_prob, log_word_prob, seq, rating_pred, rating, feature
        )

        for k, v in output.items():
            self.log_metrics(f"valid/{k}", v, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        user, item, rating, seq, feature, feature_neg, pre_pred_rating = batch
        rating_predict, tokens_test, tokens_predict, text_test, text_predict = (
            self.generate(user, item, pre_pred_rating, seq, feature)
        )
        outputs = self.get_metrics(
            rating.tolist(),
            rating_predict,
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
            feature,
            feature_neg,
        )
        return outputs

    def generate(self, user, item, pre_pred_rating, seq, feature):
        bos = seq[:, 0].unsqueeze(1)

        if self.use_feature:
            text = torch.cat([feature, bos], 1)  # (batch_size, src_len - 1)
        else:
            text = bos  # (batch_size, src_len - 1)
        start_idx = text.size(1)

        ids_predict, context_predict, rating_predict = [], [], []
        for idx in range(self.max_seq_len):
            # produce a word at each step
            if idx == 0:
                _, log_word_prob, log_context_dis, rating_pred, _ = self.forward(
                    user, item, pre_pred_rating, text, False
                )  # (batch_size, ntoken), (batch_size, ntoken), (batch_size,)
                if isinstance(rating_pred, tuple):
                    rating_pred = rating_pred[0]
                rating_predict.extend(rating_pred.tolist())
                context = self.predict(
                    log_context_dis, topk=self.max_seq_len
                )  # (batch_size, words)
                context_predict.extend(context.tolist())
            else:
                _, log_word_prob, _, _, _ = self.forward(
                    user, item, pre_pred_rating, text, False, False, False
                )  # (batch_size, ntoken)
            word_prob = log_word_prob.exp()  # (batch_size, ntoken)
            word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,)
            text = torch.cat([text, word_idx.unsqueeze(1)], 1)  # (batch_size, len++)
        ids = text[:, start_idx:].tolist()
        ids_predict.extend(ids)

        # rating
        tokens_test = [
            ids2tokens(
                ids[1:],
                self.storage.word_dict.word2idx,
                self.storage.word_dict.idx2word,
            )
            for ids in seq.tolist()
        ]
        tokens_predict = [
            ids2tokens(
                ids, self.storage.word_dict.word2idx, self.storage.word_dict.idx2word
            )
            for ids in ids_predict
        ]
        text_test = [" ".join(tokens) for tokens in tokens_test]
        text_predict = [
            " ".join([token for token in tokens if token is not None])
            for tokens in tokens_predict
        ]

        return (
            rating_predict,
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
        )

    def predict(self, log_context_dis, topk):
        word_prob = log_context_dis.exp()  # (batch_size, ntoken)
        if topk == 1:
            context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
        return context  # (batch_size, topk)

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = F.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        rating = self.base_recommender(hidden[0])  # (batch_size,)
        return rating

    def predict_sequence(self, hidden):
        word_prob = self.hidden2token(
            hidden[self.src_len :]
        )  # (tgt_len, batch_size, ntoken)
        log_word_prob = F.log_softmax(word_prob, dim=-1)
        return word_prob, log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = F.log_softmax(word_prob, dim=-1)
        return word_prob, log_word_prob

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        # NOTE: paper statement
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.opt_factor, patience=self.opt_step_size
        )
        # # NOTE: github
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, self.opt_step_size, gamma=self.opt_factor
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/loss",
        }
