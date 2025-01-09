from statistics import mean
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from models.common import BASE, MFRating, MLPRating, ids2tokens_tokenizer


class PEPLER(BASE):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        tokenizer: PreTrainedTokenizer,
        min_rating: int,
        max_rating: int,
        storage: ReviewDataLoader,
        type_rating_embedding: Optional[int] = None,
        use_seq_optimizers: bool = True,
        patience: int = 3,
        max_seq_len: int = 15,
        reg_text: float = 1.0,
        reg_rating: float = 0.1,
        pretrained_model_name: str = "gpt2",
        rec_type: Optional[str] = "mlp",
        n_hidden_layers: int = 2,
        d_hidden: int = 400,
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

        self.lm = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        self.lm.resize_token_embeddings(
            len(tokenizer)
        )  # three tokens added, update embedding table
        self.tokenizer = tokenizer
        self.type_rating_embedding = type_rating_embedding
        self.min_rating = min_rating
        self.max_rating = max_rating

        d_embed = self.lm.transformer.wte.weight.size(1)
        self.src_len = 2 if self.type_rating_embedding is None else 3
        self.max_seq_len = max_seq_len
        self.user_embeddings = nn.Embedding(n_users, d_embed)
        self.item_embeddings = nn.Embedding(n_items, d_embed)

        if self.type_rating_embedding == 0:
            self.rating_embedding = nn.Embedding(1, d_embed)
        elif self.type_rating_embedding == 1:
            self.n_rating_embedding = self.max_rating
            self.rating_embedding = nn.Embedding(self.n_rating_embedding + 1, d_embed)

        self.reg_text = reg_text
        self.reg_rating = reg_rating

        self.initialize_tokens()

        self.rec_type = rec_type
        self.recommender = None  # pepler
        if rec_type == "mf":  # pepler-mf
            self.recommender = MFRating()
        elif rec_type == "mlp":  # pepler-mpl
            self.recommender = MLPRating(
                d_embed=d_embed,
                width=2,
                n_hidden_layers=n_hidden_layers,
                d_hidden=d_hidden,
            )

        self.criterion_rating = nn.MSELoss()
        self.criterion_text = nn.CrossEntropyLoss()

        self.valid_losses = []
        self.use_seq_optimizers = use_seq_optimizers
        self.patience = patience
        self.phase = 1
        self.num_epochs_no_improvement = 0
        self.best_loss = float(torch.inf)

    def initialize_tokens(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

        if self.type_rating_embedding is not None:
            nn.init.xavier_uniform_(self.rating_embedding.weight)

    def forward(
        self,
        user,
        item,
        pre_pred_rating,
        text,
        mask,
        rating_prediction=True,
        ignore_index=-100,
    ):
        device = user.device
        batch_size = user.size(0)

        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1)], 1)

        if self.type_rating_embedding is not None:
            r_src = None
            if self.type_rating_embedding == 0:
                r_src = self.rating_embedding.weight.data.unsqueeze(0).expand(
                    batch_size, -1, -1
                ) * pre_pred_rating.unsqueeze(1).unsqueeze(2)
            elif self.type_rating_embedding == 1:
                pre_pred_rating = torch.clamp(
                    pre_pred_rating, min=self.min_rating, max=self.max_rating
                )
                pre_pred_rating = torch.round(
                    (pre_pred_rating / int(self.max_rating / self.n_rating_embedding))
                ).to(torch.int)
                r_src = self.rating_embedding(pre_pred_rating).unsqueeze(1)
            assert r_src is not None, "r_src should be included"
            src = torch.cat([src, r_src], 1)

        w_src = self.lm.transformer.wte(text)
        src = torch.cat([src, w_src], 1)

        rating = None
        if rating_prediction:
            assert self.recommender is not None, "self.recommender should be included"
            if self.rec_type == "mf":
                rating = self.recommender(u_src, i_src)
            elif self.rec_type == "mlp":
                ui_src = torch.cat([u_src, i_src], 1)
                rating = self.recommender(ui_src)

        if mask is None:
            return self.lm(inputs_embeds=src), None, rating
        else:
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(
                device
            )
            pad_input = torch.cat([pad_left, mask], 1)
            labels = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))

            return (
                self.lm(attention_mask=pad_input, inputs_embeds=src),
                labels,
                rating,
            )

    def lossfun(self, output, labels, rating_pred, rating_gt):
        # loss_t = output.loss
        shift_logits = output.logits[..., self.src_len : -1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_t = self.criterion_text(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss_r = self.criterion_rating(rating_pred, rating_gt)
        loss = self.reg_text * loss_t + self.reg_rating * loss_r
        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_r": loss_r,
        }

    def training_step(self, batch, batch_idx):
        user, item, rating, seq, mask, _, _, pre_pred_rating = batch
        print("[test] rating: ", rating)
        print("[test] pre_pred_rating: ", pre_pred_rating)

        # (test)
        if batch_idx % 500 == 0:
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, pre_pred_rating, seq
            )
            print(
                f"[test] (train) batch_idx: {batch_idx} | text: {text[0]} | text_pred: {text_pred[0]}"
            )

        output, labels, rating_pred = self.forward(
            user, item, pre_pred_rating, seq, mask
        )
        output = self.lossfun(output, labels, rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(self, batch, batch_idx):
        user, item, rating, seq, mask, _, _, pre_pred_rating = batch

        if (
            self.current_epoch % self.check_gen_text_every_n_epoch == 0
            and batch_idx == 0
        ):
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, pre_pred_rating, seq
            )
            self.log_samples(rating.tolist(), rating_pred, text, text_pred)

        output, labels, rating_pred = self.forward(
            user, item, pre_pred_rating, seq, mask
        )
        output = self.lossfun(output, labels, rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"valid/{k}", v, on_step=False)

        self.valid_losses.append(output["loss"])

    def on_validation_epoch_end(self):
        current_loss = torch.stack(self.valid_losses).mean()
        self.valid_losses.clear()

        if self.use_seq_optimizers:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.num_epochs_no_improvement = 0
            else:
                self.num_epochs_no_improvement += 1

            if self.num_epochs_no_improvement >= 2:
                if self.phase == 1:
                    for _, param in self.named_parameters():
                        param.requires_grad = True
                    self.phase = 2
                    self.num_epochs_no_improvement = 0
                    print("[test] required_grad updated!")
                elif self.phase == 2:
                    # ATTENTION: stopped by EarlyStoppingCallback.
                    if self.num_epochs_no_improvement > self.patience:
                        # TODO: include logger
                        print(
                            "[WARNING] Training should be stopped as no improvement seen during final phase."
                        )

            print(
                f"[test] current_loss: {current_loss} | self.best_loss: {self.best_loss} | self.num_epochs_no_improvement: {self.num_epochs_no_improvement} | self.patience: {self.patience}"
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        user, item, rating, seq, _, feature, feature_neg, pre_pred_rating = batch
        print("[test] rating: ", rating)
        print("[test] pre_pred_rating: ", pre_pred_rating)
        rating_predict, tokens_test, tokens_predict, text_test, text_predict = (
            self.generate(user, item, pre_pred_rating, seq)
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

    def generate(self, user, item, pre_pred_rating, seq):
        text = seq[:, 0].unsqueeze(1)  # bos, (batch_size, 1)

        ids_predict, rating_predict = [], []
        for idx in range(self.max_seq_len):
            if idx == 0:
                outputs, _, rating_pred = self.forward(
                    user, item, pre_pred_rating, text, None
                )
                assert rating_pred is not None, "rating_pred should be included"
                rating_predict.extend(rating_pred.tolist())
            else:
                outputs, _, _ = self.forward(
                    user, item, pre_pred_rating, text, None, False
                )
            last_token = outputs.logits[:, -1, :]
            word_prob = torch.softmax(last_token, dim=-1)
            token = torch.argmax(word_prob, dim=1, keepdim=True)
            text = torch.cat([text, token], 1)
        ids = text[:, 1:].tolist()
        ids_predict.extend(ids)

        tokens_test = [
            ids2tokens_tokenizer(ids[1:], self.tokenizer) for ids in seq.tolist()
        ]
        tokens_predict = [
            ids2tokens_tokenizer(ids, self.tokenizer) for ids in ids_predict
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

    def configure_optimizers(self) -> Any:
        if self.use_seq_optimizers:
            for name, param in self.named_parameters():
                if "transformer" in name:
                    param.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        return {
            "optimizer": optimizer,
            "monitor": "valid/loss",
        }
