import math
from typing import Optional

import torch
import torch.nn as nn

from loaders.helpers import ReviewDataLoader
from models.common import MLPRating, get_erra_mask, get_square_subsequent_mask
from models.peter import PETER


class ERRA(PETER):

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
        user_profile_embeds: Optional[torch.Tensor] = None,
        item_profile_embeds: Optional[torch.Tensor] = None,
        max_seq_len: int = 15,
        reg_text: float = 1.0,
        reg_context: float = 1.0,
        reg_rating: float = 0.1,
        reg_aspect: float = 0.02,
        dropout: float = 0.2,
        opt_lr: float = 0.1,
        opt_wd: float = 1e-4,
        opt_factor: float = 0.25,
        opt_step_size: int = 1,
        check_gen_text_every_n_epoch: int = 10,
        check_n_samples: int = 3,
        save_root: str = "",
    ):
        super().__init__(
            d_embed=d_embed,
            n_head=n_head,
            n_hid=n_hid,
            n_layers=n_layers,
            peter_mask=peter_mask,
            n_users=n_users,
            n_items=n_items,
            src_len=src_len,
            n_tokens=n_tokens,
            pad_idx=pad_idx,
            storage=storage,
            use_feature=True,
            max_seq_len=max_seq_len,
            reg_text=reg_text,
            reg_context=reg_context,
            reg_rating=reg_rating,
            dropout=dropout,
            opt_lr=opt_lr,
            opt_wd=opt_wd,
            opt_factor=opt_factor,
            opt_step_size=opt_step_size,
            check_gen_text_every_n_epoch=check_gen_text_every_n_epoch,
            check_n_samples=check_n_samples,
            save_root=save_root,
        )
        # (override)
        self.base_recommender = MLPRating(d_embed, 3)
        if peter_mask:
            self.attn_mask = get_erra_mask(self.src_len, self.tgt_len)
        else:
            self.attn_mask = get_square_subsequent_mask(self.src_len + self.tgt_len)

        # (addition)
        self.reg_aspect = reg_aspect
        self.user_profile_embeds = user_profile_embeds
        self.item_profile_embeds = item_profile_embeds

        self.criterion_aspect = nn.CrossEntropyLoss()

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        pre_pred_rating: torch.Tensor,
        text: torch.Tensor,
        seq_prediction: bool = True,
        context_prediction: bool = True,
        rating_prediction: bool = True,
    ):
        """
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (n_layers, batch_size, total_len, total_len)
        """
        device = user.device
        self.user_profile_embeds = self.user_profile_embeds.to(device)
        self.item_profile_embeds = self.item_profile_embeds.to(device)

        batch_size = user.size(0)
        total_len = self.uir_len + text.size(
            1
        )  # NOTE: no need +2 (feature: (text)4 -> (src)2)

        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(
            device
        )  # (total_len, total_len)
        left = (
            torch.zeros(batch_size, self.uir_len + 4).bool().to(device)
        )  # (batch_size, ui_len + aspect_len + sentence_len)
        right = (
            text[:, 4:] == self.pad_idx
        )  # replace pad_idx with True and others with False
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user)  # (batch_size, d_embed)
        i_src = self.item_embeddings(item)  # (batch_size, d_embed)
        w_src = torch.transpose(self.word_embeddings(text), 0, 1)
        # NOTE: one aspect seems to include two words (neg, pos). So avg pooling between the two words is selected.
        a1_src = torch.mean(w_src[0:2], dim=0)
        a2_src = torch.mean(w_src[2:4], dim=0)
        user_profile = self.user_profile_embeds[user]  # (batch_size, d_embed)
        item_profile = self.item_profile_embeds[item]  # (batch_size, d_embed)

        src = torch.cat(
            [
                u_src.unsqueeze(0),
                i_src.unsqueeze(0),
                a1_src.unsqueeze(0),
                a2_src.unsqueeze(0),
                user_profile.unsqueeze(0),
                item_profile.unsqueeze(0),
                w_src[4:],
            ],
            0,
        )  # (total_len, batch_size, d_embed)
        src = src * math.sqrt(self.d_embed)
        src = self.pos_encoder(src)

        hidden, attns = self.transformer_encoder(
            src, attn_mask, key_padding_mask
        )  # (total_len, batch_size, d_embed), (n_layers, batch_size, total_len_tgt, total_len_src)

        rating, log_context_dis = None, None
        if rating_prediction:
            rating = self.predict_rating(hidden, u_src, i_src)  # (batch_size,)
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

        # NOTE: official implementation does not include aspect loss
        # see also: https://github.com/Complex-data/ERRA/blob/main/main.py#L197
        # Also, authors set reg_aspect with small value (0.02). -> We ignore it?
        (len_seq, batch_size, n_words) = log_word_prob.size()
        bool_gt_asp = torch.zeros(batch_size, n_words, device=log_word_prob.device)
        bool_gt_asp.scatter_(1, asp[:, :4], 1).to(log_word_prob.device)
        indices_per_sample = torch.arange(batch_size).unsqueeze(1)
        expanded_indices = indices_per_sample.repeat(1, len_seq).view(-1)
        bool_gt_asp = bool_gt_asp[expanded_indices]
        loss_a = self.criterion_aspect(word_prob.view(-1, n_words), bool_gt_asp)

        loss = (
            self.reg_text * loss_t
            + self.reg_context * loss_c
            + self.reg_rating * loss_r
            + self.reg_aspect * loss_a
        )

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_c": loss_c,
            "loss_r": loss_r,
            "loss_a": loss_a,
        }

    def training_step(self, batch, batch_idx):
        user, item, rating, seq, _, _, aspect = batch

        # (test)
        if batch_idx % 1000 == 0:
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, None, seq, aspect
            )
            print(
                f"[test] (train) batch_idx: {batch_idx} | text: {text[0]} | text_pred: {text_pred[0]}"
            )
        if self.use_feature:
            text = torch.cat(
                [aspect, seq[:, :-1]], 1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        word_prob, log_word_prob, log_context_dis, rating_pred, _ = self.forward(
            user, item, None, text
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis, word_prob, log_word_prob, seq, rating_pred, rating, aspect
        )

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(self, batch, batch_idx):
        user, item, rating, seq, _, _, aspect = batch

        if (
            self.current_epoch % self.check_gen_text_every_n_epoch == 0
            and batch_idx == 0
        ):
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, None, seq, aspect
            )
            self.log_samples(rating.tolist(), rating_pred, text, text_pred)

        if self.use_feature:
            text = torch.cat(
                [aspect, seq[:, :-1]], 1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        word_prob, log_word_prob, log_context_dis, rating_pred, _ = self.forward(
            user, item, None, text
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis, word_prob, log_word_prob, seq, rating_pred, rating, aspect
        )

        for k, v in output.items():
            self.log_metrics(f"valid/{k}", v, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        user, item, rating, seq, feature, feature_neg, aspect = batch
        rating_predict, tokens_test, tokens_predict, text_test, text_predict = (
            self.generate(user, item, None, seq, aspect)
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

    def predict_rating(self, hidden, u_src, i_src):
        hal = torch.cat([hidden[1], u_src, i_src], dim=1)
        rating = self.base_recommender(hal)
        return rating

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.opt_factor, patience=self.opt_step_size
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/loss",
        }
