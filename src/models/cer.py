from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.common import MLPRating
from models.peter import PETER


class CER(PETER):
    """Causal Explainable Recommendation (CER) model.

    This class implements the CER model which extends the PETER model by adding
    an additional rating predictor that uses the generated explanation to predict ratings.
    This creates a causal relationship between explanations and ratings.

    The model enhances the base PETER architecture by:
    - Adding an additional MLP-based rating predictor for explanations
    - Creating a causal link between explanations and ratings
    - Using explanation-based rating prediction for consistency

    Attributes:
        additional_recommender (MLPRating): Additional MLP-based rating predictor for explanations
    """

    def __init__(self, *args, **kwargs):
        """Initialize the CER model.

        Args:
            *args: Variable length argument list passed to PETER model
            **kwargs: Arbitrary keyword arguments passed to PETER model
        """
        super().__init__(*args, **kwargs)
        self.additional_recommender = MLPRating(self.d_embed)

    def predict_rating(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ratings using both base and explanation-based predictors.

        Args:
            hidden (torch.Tensor): Hidden state tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - rating (torch.Tensor): Base rating prediction
                - explanation_rating (torch.Tensor): Explanation-based rating prediction
        """
        rating = self.base_recommender(hidden[0])  # (batch_size,)
        hidden_explanation, _ = hidden[self.src_len :].max(0)
        explanation_rating = self.additional_recommender(hidden_explanation)
        return rating, explanation_rating

    def lossfun(
        self,
        log_context_dis: torch.Tensor,
        word_prob: torch.Tensor,
        log_word_prob: torch.Tensor,
        seq: torch.Tensor,
        rating_pred: Tuple[torch.Tensor, torch.Tensor],
        rating: torch.Tensor,
        asp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculate the loss for the model.

        Args:
            log_context_dis (torch.Tensor): Log context distribution
            word_prob (torch.Tensor): Word probabilities
            log_word_prob (torch.Tensor): Log word probabilities
            seq (torch.Tensor): Input sequence tensor
            rating_pred (Tuple[torch.Tensor, torch.Tensor]): Tuple of base and explanation ratings
            rating (torch.Tensor): Ground truth ratings
            asp (Optional[torch.Tensor], optional): Aspect tensor. Defaults to None

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing different loss components
        """
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
        loss_r = self.criterion_rating(rating, rating_pred[0])
        loss_r_exp = self.criterion_rating(rating_pred[0], rating_pred[1])
        loss = (
            self.reg_text * loss_t
            + self.reg_context * loss_c
            + self.reg_rating * loss_r
            + self.reg_rating * loss_r_exp
        )

        return {
            "loss": loss,
            "loss_t": loss_t,
            "loss_c": loss_t,
            "loss_r": loss_r,
            "loss_r_exp": loss_r_exp,
        }
