import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from models.common import BASE, ids2tokens_tokenizer

logger = logging.getLogger(__name__)


class PEPLER_D(BASE):
    """Personalized Explanation Generation with Pre-trained Language Model for Recommendation (Discriminative Version).

    This class implements the discriminative version of the PEPLER model which uses a pre-trained
    language model (GPT-2) to generate explanations based on user-item features.

    The model focuses on generating explanations without explicit rating prediction,
    using pre-computed user-item features as input.

    Attributes:
        lm (GPT2LMHeadModel): Pre-trained GPT-2 language model
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing
        max_seq_len (int): Maximum sequence length for generation
    """

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
        """Initialize the PEPLER_D model.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for text processing
            storage (ReviewDataLoader): Data loader for reviews
            max_seq_len (int, optional): Maximum sequence length. Defaults to 15
            n_keywords (int, optional): Number of keywords to use. Defaults to 3
            pretrained_model_name (str, optional): Name of pretrained model. Defaults to "gpt2"
            opt_lr (float, optional): Learning rate. Defaults to 0.1
            opt_wd (float, optional): Weight decay. Defaults to 0.0
            opt_factor (float, optional): Learning rate reduction factor. Defaults to 0.25
            opt_step_size (int, optional): Steps between learning rate updates. Defaults to 1
            check_gen_text_every_n_epoch (int, optional): Epochs between text generation checks. Defaults to 10
            check_n_samples (int, optional): Number of samples to check. Defaults to 3
            save_root (str, optional): Directory to save model outputs. Defaults to ""
        """
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

    def forward(
        self,
        feat_ui: torch.Tensor,
        mask_feat_ui: torch.Tensor,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor],
        ignore_index: int = -100,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Forward pass of the PEPLER_D model.

        Args:
            feat_ui (torch.Tensor): User-item feature tensor
            mask_feat_ui (torch.Tensor): Mask for user-item features
            seq (torch.Tensor): Input sequence tensor
            mask (Optional[torch.Tensor]): Attention mask
            ignore_index (int, optional): Index to ignore in loss calculation. Defaults to -100

        Returns:
            tuple: A tuple containing:
                - model_output (Any): Output from the language model
                - labels (Optional[torch.Tensor]): Labels for masked tokens
        """
        device = feat_ui.device
        text = torch.cat([feat_ui, seq], 1)  # (batch_size, total_len)
        src = self.lm.transformer.wte(text)  # (batch_size, total_len, emsize)

        if mask is None:
            mask = torch.ones_like(seq, device=device)
            pad_input = torch.cat([mask_feat_ui, mask], 1)
            return self.lm(inputs_embeds=src), None
        else:
            pad_input = torch.cat([mask_feat_ui, mask], 1)
            labels = torch.where(
                mask == 1, seq, torch.tensor(ignore_index).to(device)
            )
            return self.lm(attention_mask=pad_input, inputs_embeds=src), labels

    def lossfun(
        self, output: Any, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate the loss for the model.

        Args:
            output (Any): Output from the language model
            labels (torch.Tensor): Ground truth labels

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value
        """
        shift_logits = output.logits[..., -labels.size(1) : -1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.criterion_text(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"loss": loss}

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of training data
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Training loss
        """
        feat_ui, mask_feat_ui, _, seq, mask, _, _ = batch

        # (test)
        if batch_idx % 500 == 0:
            _, _, text, text_pred = self.generate(feat_ui, mask_feat_ui, seq)
            logger.info(
                "[test] (train) batch_idx: %s | text: %s | text_pred: %s",
                batch_idx,
                text[0],
                text_pred[0],
            )

        output, labels = self.forward(feat_ui, mask_feat_ui, seq, mask)
        output = self.lossfun(output, labels)

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        """Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of validation data
            batch_idx (int): Index of the current batch
        """
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

    def test_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Union[float, List[str]]]:
        """Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of test data
            batch_idx (int): Index of the current batch
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary containing test metrics
        """
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

    def generate(
        self,
        feat_ui: torch.Tensor,
        mask_feat_ui: torch.Tensor,
        seq: torch.Tensor,
    ) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
        """Generate explanatory text based on user-item features.

        Args:
            feat_ui (torch.Tensor): User-item feature tensor
            mask_feat_ui (torch.Tensor): Mask for user-item features
            seq (torch.Tensor): Input sequence tensor

        Returns:
            Tuple containing:
                - tokens_test (List[List[str]]): Original text tokens
                - tokens_predict (List[List[str]]): Generated text tokens
                - text_test (List[str]): Original text strings
                - text_predict (List[str]): Generated text strings
        """
        _, src_len = feat_ui.size()
        idxs = seq[:, 0].unsqueeze(1)  # bos, (batch_size, 1)
        idxs_predict = []

        for _ in range(self.max_seq_len):
            outputs, _ = self.forward(feat_ui, mask_feat_ui, idxs, None)
            last_token = outputs.logits[:, -1, :]
            word_prob = torch.softmax(last_token, dim=-1)
            idx = torch.argmax(
                word_prob, dim=1, keepdim=True
            )  # (batch_size, 1)
            idxs = torch.cat([idxs, idx], 1)  # (batch_size, len++)
        idxs = idxs[:, src_len:].tolist()  # (batch_size, seq_len)
        idxs_predict.extend(idxs)

        tokens_test = [
            ids2tokens_tokenizer(ids_seq[1:], self.tokenizer)
            for ids_seq in seq.tolist()
        ]
        tokens_predict = [
            ids2tokens_tokenizer(ids_seq, self.tokenizer)
            for ids_seq in idxs_predict
        ]
        text_test = [" ".join(tokens) for tokens in tokens_test]
        text_predict = [
            " ".join([token for token in tokens if token is not None])
            for tokens in tokens_predict
        ]

        return tokens_test, tokens_predict, text_test, text_predict

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers for training.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        return {
            "optimizer": optimizer,
            "monitor": "valid/loss",
        }
