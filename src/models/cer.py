from models.common import MLPRating
from models.peter import PETER


class CER(PETER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_recommender = MLPRating(self.d_embed)

    def predict_rating(self, hidden):
        rating = self.base_recommender(hidden[0])  # (batch_size,)
        hidden_explanation, _ = hidden[self.src_len :].max(0)
        explanation_rating = self.additional_recommender(hidden_explanation)
        return rating, explanation_rating

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
