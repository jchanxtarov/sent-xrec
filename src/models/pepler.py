import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from loaders.helpers import ReviewDataLoader
from models.common import BASE, MFRating, MLPRating, ids2tokens_tokenizer


class PEPLER(BASE):
    """Personalized Explanation Generation with Pre-trained Language Model for Recommendation.

    This class implements the PEPLER model which combines a pre-trained language model (GPT-2)
    with personalized recommendation capabilities to generate natural language explanations.

    The model can operate in different modes based on the recommender type:
    - PEPLER-MF: Using matrix factorization for recommendation
    - PEPLER-MLP: Using multi-layer perceptron for recommendation

    Attributes:
        lm (GPT2LMHeadModel): Pre-trained GPT-2 language model
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing
        type_rating_embedding (Optional[int]): Type of rating embedding used
        rec_type (Optional[str]): Type of recommender ('mf' or 'mlp')
        use_seq_optimizers (bool): Whether to use sequential optimizers
    """

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
        custom_logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PEPLER model.

        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset
            tokenizer (PreTrainedTokenizer): Tokenizer for text processing
            min_rating (int): Minimum rating value
            max_rating (int): Maximum rating value
            storage (ReviewDataLoader): Data loader for reviews
            type_rating_embedding (Optional[int], optional): Type of rating embedding. Defaults to None
            use_seq_optimizers (bool, optional): Whether to use sequential optimizers. Defaults to True
            patience (int, optional): Patience for early stopping. Defaults to 3
            max_seq_len (int, optional): Maximum sequence length. Defaults to 15
            reg_text (float, optional): Text regularization weight. Defaults to 1.0
            reg_rating (float, optional): Rating regularization weight. Defaults to 0.1
            pretrained_model_name (str, optional): Name of pretrained model. Defaults to "gpt2"
            rec_type (Optional[str], optional): Type of recommender. Defaults to "mlp"
            n_hidden_layers (int, optional): Number of hidden layers. Defaults to 2
            d_hidden (int, optional): Hidden dimension size. Defaults to 400
            opt_lr (float, optional): Learning rate. Defaults to 0.1
            opt_wd (float, optional): Weight decay. Defaults to 0.0
            opt_factor (float, optional): Learning rate reduction factor. Defaults to 0.25
            opt_step_size (int, optional): Steps between learning rate updates. Defaults to 1
            check_gen_text_every_n_epoch (int, optional): Epochs between text generation checks. Defaults to 10
            check_n_samples (int, optional): Number of samples to check. Defaults to 3
            save_root (str, optional): Directory to save model outputs. Defaults to ""
            custom_logger (Optional[logging.Logger], optional): Custom logger instance to use. If None, uses default logger. Defaults to None
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
            custom_logger,
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
            self.rating_embedding = nn.Embedding(
                self.n_rating_embedding + 1, d_embed
            )

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

        self.valid_losses: List[torch.Tensor] = []
        self.use_seq_optimizers = use_seq_optimizers
        self.patience = patience
        self.phase = 1
        self.num_epochs_no_improvement = 0
        self.best_loss = float(torch.inf)

    def initialize_tokens(self) -> None:
        """Initialize the embedding weights for users, items, and ratings."""
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

        if self.type_rating_embedding is not None:
            nn.init.xavier_uniform_(self.rating_embedding.weight)

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        pre_pred_rating: torch.Tensor,
        text: torch.Tensor,
        mask: Optional[torch.Tensor],
        rating_prediction: bool = True,
        ignore_index: int = -100,
    ) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the PEPLER model.

        Args:
            user (torch.Tensor): User indices tensor
            item (torch.Tensor): Item indices tensor
            pre_pred_rating (torch.Tensor): Predicted ratings from previous step
            text (torch.Tensor): Input text sequences
            mask (Optional[torch.Tensor]): Attention mask
            rating_prediction (bool, optional): Whether to predict ratings. Defaults to True
            ignore_index (int, optional): Index to ignore in loss calculation. Defaults to -100

        Returns:
            tuple: A tuple containing:
                - model_output (Any): Output from the language model
                - labels (Optional[torch.Tensor]): Labels for masked tokens
                - rating (Optional[torch.Tensor]): Predicted ratings
        """
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
                    (
                        pre_pred_rating
                        / int(self.max_rating / self.n_rating_embedding)
                    )
                ).to(torch.int)
                r_src = self.rating_embedding(pre_pred_rating).unsqueeze(1)
            assert r_src is not None, "r_src should be included"
            src = torch.cat([src, r_src], 1)

        w_src = self.lm.transformer.wte(text)
        src = torch.cat([src, w_src], 1)

        rating = None
        if rating_prediction:
            assert (
                self.recommender is not None
            ), "self.recommender should be included"
            if self.rec_type == "mf":
                rating = self.recommender(u_src, i_src)
            elif self.rec_type == "mlp":
                ui_src = torch.cat([u_src, i_src], 1)
                rating = self.recommender(ui_src)

        if mask is None:
            return self.lm(inputs_embeds=src), None, rating
        else:
            pad_left = torch.ones(
                (batch_size, self.src_len), dtype=torch.int64
            ).to(device)
            pad_input = torch.cat([pad_left, mask], 1)
            labels = torch.where(
                mask == 1,
                text,
                torch.tensor(ignore_index).to(device),
            )
            return (
                self.lm(attention_mask=pad_input, inputs_embeds=src),
                labels,
                rating,
            )

    def lossfun(
        self,
        output: Any,
        labels: torch.Tensor,
        rating_pred: torch.Tensor,
        rating_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss for the model.

        Args:
            output (Any): Output from the language model
            labels (torch.Tensor): Ground truth labels
            rating_pred (torch.Tensor): Predicted ratings
            rating_gt (torch.Tensor): Ground truth ratings

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing different loss components
        """
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

    def training_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of training data
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Training loss
        """
        user, item, rating, seq, mask, _, _, pre_pred_rating = batch
        self._custom_logger.debug("[test] rating: %s", rating)
        self._custom_logger.debug(
            "[test] pre_pred_rating: %s", pre_pred_rating
        )

        # Diagnostic text generation
        if batch_idx % 500 == 0:
            rating_pred, _, _, text, text_pred = self.generate(
                user, item, pre_pred_rating, seq
            )
            self._custom_logger.info(
                "[test] (train) batch_idx: %s | text: %s | text_pred: %s",
                batch_idx,
                text[0],
                text_pred[0],
            )

        output, labels, rating_pred = self.forward(
            user, item, pre_pred_rating, seq, mask
        )
        output = self.lossfun(output, labels, rating_pred, rating)

        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of validation data
            batch_idx (int): Index of the current batch
        """
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

    def on_validation_epoch_end(self) -> None:
        """
        Handle the end of validation epoch.

        Updates the model phase and handles early stopping based on validation loss.
        """
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
                    self._custom_logger.info("[test] required_grad updated!")
                elif self.phase == 2:
                    # Stopped by an EarlyStoppingCallback if implemented externally
                    if self.num_epochs_no_improvement > self.patience:
                        self._custom_logger.warning(
                            "[WARNING] Training should be stopped as no improvement "
                            "seen during final phase."
                        )

            self._custom_logger.info(
                "[test] current_loss: %s | self.best_loss: %s | self.num_epochs_no_improvement: %s | self.patience: %s",
                current_loss,
                self.best_loss,
                self.num_epochs_no_improvement,
                self.patience,
            )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of test data
            batch_idx (int): Index of the current batch
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary containing test metrics
        """
        user, item, rating, seq, _, feature, feature_neg, pre_pred_rating = (
            batch
        )
        self._custom_logger.debug("[test] rating: %s", rating)
        self._custom_logger.debug(
            "[test] pre_pred_rating: %s", pre_pred_rating
        )

        (
            rating_predict,
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
        ) = self.generate(user, item, pre_pred_rating, seq)

        outputs = self.get_metrics(
            user.tolist(),
            item.tolist(),
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

    def generate(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        pre_pred_rating: torch.Tensor,
        seq: torch.Tensor,
    ) -> Tuple[
        List[float],
        List[List[str]],
        List[List[str]],
        List[str],
        List[str],
    ]:
        """
        Generate explanatory text for recommendations.

        Args:
            user (torch.Tensor): User indices tensor
            item (torch.Tensor): Item indices tensor
            pre_pred_rating (torch.Tensor): Predicted ratings from previous step
            seq (torch.Tensor): Input sequence tensor

        Returns:
            Tuple containing:
                - rating_predict (List[float]): List of predicted ratings
                - tokens_test (List[List[str]]): Original text tokens
                - tokens_predict (List[List[str]]): Generated text tokens
                - text_test (List[str]): Original text strings
                - text_predict (List[str]): Generated text strings
        """
        text = seq[:, 0].unsqueeze(1)  # bos, (batch_size, 1)

        ids_predict, rating_predict = [], []
        for idx in range(self.max_seq_len):
            if idx == 0:
                outputs, _, rating_pred = self.forward(
                    user, item, pre_pred_rating, text, None
                )
                assert (
                    rating_pred is not None
                ), "rating_pred should be included"
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
            ids2tokens_tokenizer(ids_seq[1:], self.tokenizer)
            for ids_seq in seq.tolist()
        ]
        tokens_predict = [
            ids2tokens_tokenizer(ids_seq, self.tokenizer)
            for ids_seq in ids_predict
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

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers for training.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer configuration
        """
        if self.use_seq_optimizers:
            # Freeze GPT-2 parameters initially
            for name, param in self.named_parameters():
                if "transformer" in name:
                    param.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_lr,
            weight_decay=self.opt_wd,
        )
        return {
            "optimizer": optimizer,
            "monitor": "valid/loss",
        }
