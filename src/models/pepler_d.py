from statistics import mean
from typing import Any, Dict

import torch
from torch import nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from models.common import BASE, ids2tokens_tokenizer


class PEPLER_D(BASE):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        storage: ReviewDataLoader,
        max_seq_len: int = 15,
        n_keywords: int = 3,
        pretrained_model_name: str = "gpt2",
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
        self.lm.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.criterion_text = nn.CrossEntropyLoss()

    def forward(self, feat_ui, mask_feat_ui, seq, mask, ignore_index=-100):
        device = feat_ui.device
        text = torch.cat([feat_ui, seq], 1)  # (batch_size, total_len)
        src = self.lm.transformer.wte(text)  # (batch_size, total_len, emsize)

        if mask is None:
            mask = torch.ones_like(seq, device=device)
            pad_input = torch.cat([mask_feat_ui, mask], 1)
            return self.lm(inputs_embeds=src), None
        else:
            pad_input = torch.cat([mask_feat_ui, mask], 1)
            labels = torch.where(mask == 1, seq, torch.tensor(ignore_index).to(device))
            return self.lm(attention_mask=pad_input, inputs_embeds=src), labels

    def lossfun(self, output, labels):
        # loss = output.loss
        shift_logits = output.logits[..., -labels.size(1) : -1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.criterion_text(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        feat_ui, mask_feat_ui, _, seq, mask, _, _ = batch

        # (test)
        if batch_idx % 500 == 0:
            _, _, text, text_pred = self.generate(feat_ui, mask_feat_ui, seq)
            print(
                f"[test] (train) batch_idx: {batch_idx} | text: {text[0]} | text_pred: {text_pred[0]}"
            )

        output, labels = self.forward(feat_ui, mask_feat_ui, seq, mask)
        output = self.lossfun(output, labels)

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(self, batch, batch_idx):
        feat_ui, mask_feat_ui, rating, seq, mask, _, _ = batch

        if (
            self.current_epoch % self.check_gen_text_every_n_epoch == 0
            and batch_idx == 0
        ):
            _, _, text, text_pred = self.generate(feat_ui, mask_feat_ui, seq)
            self.log_samples(rating.tolist(), None, text, text_pred)

        output, labels = self.forward(feat_ui, mask_feat_ui, seq, mask)
        output = self.lossfun(output, labels)

        for k, v in output.items():
            self.log_metrics(f"valid/{k}", v, on_step=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        feat_ui, mask_feat_ui, _, seq, _, fea, pos_neg = batch
        tokens_test, tokens_predict, text_test, text_predict = self.generate(
            feat_ui, mask_feat_ui, seq
        )
        outputs = self.get_metrics(
            None,
            None,
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
            fea,
            pos_neg,
        )
        return outputs

    def generate(self, feat_ui, mask_feat_ui, seq):
        _, src_len = feat_ui.size()
        idxs = seq[:, 0].unsqueeze(1)  # bos, (batch_size, 1)
        idxs_predict = []
        for _ in range(self.max_seq_len):
            outputs, _ = self.forward(feat_ui, mask_feat_ui, idxs, None)
            last_token = outputs.logits[:, -1, :]
            word_prob = torch.softmax(last_token, dim=-1)
            idx = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
            idxs = torch.cat([idxs, idx], 1)  # (batch_size, len++)
        idxs = idxs[:, src_len:].tolist()  # (batch_size, seq_len)
        idxs_predict.extend(idxs)

        tokens_test = [
            ids2tokens_tokenizer(idxs[1:], self.tokenizer) for idxs in seq.tolist()
        ]
        tokens_predict = [
            ids2tokens_tokenizer(idxs, self.tokenizer) for idxs in idxs_predict
        ]
        text_test = [" ".join(tokens) for tokens in tokens_test]
        text_predict = [
            " ".join([token for token in tokens if token is not None])
            for tokens in tokens_predict
        ]

        return (
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        return {
            "optimizer": optimizer,
            "monitor": "valid/loss",
        }
