import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from loaders.helpers import ReviewDataLoader
from models.common import MLPRating, get_erra_mask, get_square_subsequent_mask
from models.peter import PETER


class ERRA(PETER):
    """Explainable Recommendation with Reciprocal Attention (ERRA) model.

    This class implements the ERRA model which extends the PETER model by incorporating
    reciprocal attention mechanisms and aspect-aware features for better explainable recommendations.

    The model enhances the base PETER architecture by:
    - Adding user and item profile embeddings
    - Incorporating aspect-aware attention
    - Using reciprocal attention for better user-item interaction modeling

    Attributes:
        user_profile_embeds (torch.Tensor): Pre-trained user profile embeddings
        item_profile_embeds (torch.Tensor): Pre-trained item profile embeddings
        reg_aspect (float): Regularization weight for aspect loss
        base_recommender (MLPRating): MLP-based rating predictor
    """

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
        user_profile_embeds: torch.Tensor,
        item_profile_embeds: torch.Tensor,
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
        """
        Initialize the ERRA model.

        Args:
            d_embed (int): Dimension of embeddings.
            n_head (int): Number of attention heads.
            n_hid (int): Hidden dimension size.
            n_layers (int): Number of transformer layers.
            peter_mask (bool): Whether to use PETER-specific attention masking.
            n_users (int): Number of users in the dataset.
            n_items (int): Number of items in the dataset.
            src_len (int): Maximum source sequence length.
            n_tokens (int): Size of the vocabulary.
            pad_idx (int): Index used for padding.
            storage (ReviewDataLoader): Data loader for reviews.
            user_profile_embeds (torch.Tensor): Pre-trained user profile embeddings.
            item_profile_embeds (torch.Tensor): Pre-trained item profile embeddings.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 15.
            reg_text (float, optional): Text regularization weight. Defaults to 1.0.
            reg_context (float, optional): Context regularization weight. Defaults to 1.0.
            reg_rating (float, optional): Rating regularization weight. Defaults to 0.1.
            reg_aspect (float, optional): Aspect regularization weight. Defaults to 0.02.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            opt_lr (float, optional): Learning rate. Defaults to 0.1.
            opt_wd (float, optional): Weight decay. Defaults to 1e-4.
            opt_factor (float, optional): Learning rate reduction factor. Defaults to 0.25.
            opt_step_size (int, optional): Steps between learning rate updates. Defaults to 1.
            check_gen_text_every_n_epoch (int, optional): Epochs between text generation checks. Defaults to 10.
            check_n_samples (int, optional): Number of samples to check. Defaults to 3.
            save_root (str, optional): Directory to save model outputs. Defaults to "".
        """
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

        # Override the base recommender
        self.base_recommender = MLPRating(d_embed, 3)

        # Choose the appropriate attention mask
        if peter_mask:
            self.attn_mask = get_erra_mask(self.src_len, self.tgt_len)
        else:
            self.attn_mask = get_square_subsequent_mask(
                self.src_len + self.tgt_len
            )

        self.reg_aspect = reg_aspect
        self.user_profile_embeds = user_profile_embeds
        self.item_profile_embeds = item_profile_embeds

        self.criterion_aspect = nn.CrossEntropyLoss()

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        pre_pred_rating: Optional[torch.Tensor],  # Not Used
        text: torch.Tensor,
        seq_prediction: bool = True,
        context_prediction: bool = True,
        rating_prediction: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Forward pass of the ERRA model.

        Args:
            user (torch.Tensor): User indices tensor.
            item (torch.Tensor): Item indices tensor.
            pre_pred_rating (Optional[torch.Tensor], optional): Pre-predicted ratings.
            text (torch.Tensor): Input text sequences.
            seq_prediction (bool, optional): Whether to predict sequences. Defaults to True.
            context_prediction (bool, optional): Whether to predict context. Defaults to True.
            rating_prediction (bool, optional): Whether to predict ratings. Defaults to True.

        Returns:
            tuple:
                - word_prob (torch.Tensor): Word probabilities (tgt_len, batch_size, n_tokens).
                - log_word_prob (torch.Tensor): Log word probabilities (tgt_len, batch_size, n_tokens).
                - log_context_dis (Optional[torch.Tensor]): Log context distribution per batch if `context_prediction` is True.
                - rating (Optional[torch.Tensor]): Predicted ratings if `rating_prediction` is True.
                - attns (torch.Tensor): Attention weights from the transformer encoder.
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
        key_padding_mask = torch.cat(
            [left, right], 1
        )  # (batch_size, total_len)

        # Embeddings
        u_src = self.user_embeddings(user)  # (batch_size, d_embed)
        i_src = self.item_embeddings(item)  # (batch_size, d_embed)
        w_src = torch.transpose(self.word_embeddings(text), 0, 1)
        # NOTE: one aspect seems to include two words (neg, pos). So avg pooling between the two words is selected.
        a1_src = torch.mean(w_src[0:2], dim=0)
        a2_src = torch.mean(w_src[2:4], dim=0)

        # User profile and item profile
        user_profile = self.user_profile_embeds[user]  # (batch_size, d_embed)
        item_profile = self.item_profile_embeds[item]  # (batch_size, d_embed)

        # Concatenate embeddings for user, item, two aspects, user_profile, item_profile, and rest of the words
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
            dim=0,
        )  # (total_len, batch_size, d_embed)

        # Scale and position-encode
        src = src * math.sqrt(self.d_embed)
        src = self.pos_encoder(src)

        # Pass through transformer encoder
        hidden, attns = self.transformer_encoder(
            src, attn_mask, key_padding_mask
        )  # (total_len, batch_size, d_embed), (n_layers, batch_size, total_len_tgt, total_len_src)

        rating, log_context_dis = None, None
        if rating_prediction:
            # Concatenate item hidden + user emb + item emb
            hid = torch.cat([hidden[1], u_src, i_src], dim=1)
            rating = self.predict_rating(hid)  # (batch_size,)

        # Context prediction
        if context_prediction:
            log_context_dis = self.predict_context(
                hidden
            )  # (batch_size, ntoken)

        # Sequence prediction
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
        log_context_dis: torch.Tensor,
        word_prob: torch.Tensor,
        log_word_prob: torch.Tensor,
        seq: torch.Tensor,
        rating_pred: torch.Tensor,
        rating: torch.Tensor,
        asp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss for the model.

        Args:
            log_context_dis (torch.Tensor): Log context distribution of shape (batch_size, n_tokens).
            word_prob (torch.Tensor): Word probabilities of shape (tgt_len, batch_size, n_tokens).
            log_word_prob (torch.Tensor): Log word probabilities of shape (tgt_len, batch_size, n_tokens).
            seq (torch.Tensor): Ground truth sequence tensor of shape (batch_size, seq_len).
            rating_pred (torch.Tensor): Predicted ratings of shape (batch_size,).
            rating (torch.Tensor): Ground truth ratings of shape (batch_size,).
            asp (Optional[torch.Tensor]): Aspect tensor. Must not be None for ERRA.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss components:
                - "loss": Total combined loss
                - "loss_t": Text generation loss
                - "loss_c": Context prediction loss
                - "loss_r": Rating prediction loss
                - "loss_a": Aspect loss
        """
        assert asp is not None, "asp should be included"

        # Expand context distribution to match sequence dimension
        # (batch_size, n_tokens) -> (tgt_len - 1, batch_size, n_tokens)
        context_dis = log_context_dis.unsqueeze(0).repeat(
            (self.tgt_len - 1, 1, 1)
        )  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        loss_t = self.criterion_text(
            log_word_prob.view(-1, self.n_tokens),
            seq[:, 1:].permute(1, 0).reshape((-1,)),
        )

        # Context prediction loss
        loss_c = self.criterion_text(
            context_dis.view(-1, self.n_tokens),
            seq[:, 1:-1].permute(1, 0).reshape((-1,)),
        )

        # Rating prediction loss
        loss_r = self.criterion_rating(rating_pred, rating)

        # NOTE: official implementation does not include aspect loss
        # see also: https://github.com/Complex-data/ERRA/blob/main/main.py#L197
        # Also, authors set reg_aspect with small value (0.02). -> We ignore it?
        (len_seq, batch_size, n_words) = log_word_prob.size()
        bool_gt_asp = torch.zeros(
            batch_size, n_words, device=log_word_prob.device
        )
        bool_gt_asp.scatter_(1, asp[:, :4], 1).to(log_word_prob.device)
        indices_per_sample = torch.arange(batch_size).unsqueeze(1)
        expanded_indices = indices_per_sample.repeat(1, len_seq).view(-1)
        bool_gt_asp = bool_gt_asp[expanded_indices]

        loss_a = self.criterion_aspect(
            word_prob.view(-1, n_words), bool_gt_asp
        )

        # Weighted sum of all losses
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

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss.
        """
        user, item, rating, seq, _, _, aspect = batch

        # (test) occasionally generate text for quick check
        if batch_idx % 1000 == 0:
            rating_pred, _, _, text_gt, text_pred = self.generate(
                user, item, None, seq, aspect
            )
            print(
                f"[test] (train) batch_idx: {batch_idx} | "
                f"text: {text_gt[0]} | text_pred: {text_pred[0]}"
            )

        # If feature usage is enabled, prepend aspect tokens
        if self.use_feature:
            text = torch.cat(
                [aspect, seq[:, :-1]], 1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        word_prob, log_word_prob, log_context_dis, rating_pred, _ = (
            self.forward(user, item, None, text)
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis,
            word_prob,
            log_word_prob,
            seq,
            rating_pred,
            rating,
            aspect,
        )

        # Log metrics
        for k, v in output.items():
            self.log_metrics(f"train/{k}", v, on_step=True)

        return output["loss"].to(torch.float32)

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of validation data.
            batch_idx (int): Index of the current batch.
        """
        user, item, rating, seq, _, _, aspect = batch

        # Generate text samples if it's time to check
        if (
            self.current_epoch % self.check_gen_text_every_n_epoch == 0
            and batch_idx == 0
        ):
            rating_pred, _, _, text_gt, text_pred = self.generate(
                user, item, None, seq, aspect
            )
            self.log_samples(rating.tolist(), rating_pred, text_gt, text_pred)

        if self.use_feature:
            text = torch.cat(
                [aspect, seq[:, :-1]], dim=1
            )  # (batch_size, src_len + tgt_len - 2)
        else:
            text = seq[:, :-1]  # (batch_size, src_len + tgt_len - 2)

        word_prob, log_word_prob, log_context_dis, rating_pred, _ = (
            self.forward(user, item, None, text)
        )  # (tgt_len, batch_size, ntoken), (batch_size, ntoken), (batch_size,)
        output = self.lossfun(
            log_context_dis,
            word_prob,
            log_word_prob,
            seq,
            rating_pred,
            rating,
            aspect,
        )

        for k, v in output.items():
            self.log_metrics(f"valid/{k}", v, on_step=False)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of test data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary containing test metrics.
        """
        user, item, rating, seq, feature, feature_neg, aspect = batch
        (
            rating_predict,
            tokens_test,
            tokens_predict,
            text_test,
            text_predict,
        ) = self.generate(user, item, None, seq, aspect)
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

    def predict_rating(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict rating from hidden states.

        Args:
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, N).

        Returns:
            torch.Tensor: Predicted rating values of shape (batch_size,).
        """
        return self.base_recommender(hidden)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler configuration.
        """
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.opt_factor,
            patience=self.opt_step_size,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/loss",
        }
