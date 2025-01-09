import logging
import os
from collections import Counter
from typing import Optional

import pandas as pd


def data_loader(
    logger: logging.Logger,
    dataset: str,
    max_seq_len: int,
    max_data_size: Optional[int] = None,
) -> dict:

    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    FILE_PATH = os.path.join(base_path, f"datasets/{dataset}_exps.pkl.gz")

    df = pd.read_pickle(FILE_PATH, compression="gzip")

    if max_data_size is not None:
        df = df[:max_data_size]
        logger.warning(f"*** Development mode with max_data_size = {max_data_size} ***")
        logger.info(f"(After deletion using max_data_size) #reviews: {len(df)}")
        df["role"] = 0
        df["index"] = [i for i in range(len(df))]
        df.loc[df.groupby("user").index.idxmax(), "role"] = 2
        _df_except_test = df.drop(df.loc[df.groupby("user").index.idxmax()].index)
        df.loc[_df_except_test.groupby("user").index.idxmax(), "role"] = 1
        del _df_except_test

    n_users = len(set(df["user"]))
    n_items = len(set(df["item"]))
    n_features_pos = len(set([tmp[0] for tmp in list(df["template"])]))
    n_features_neg = len(set([tmp[1] for tmp in list(df["template"])]))
    logger.info(
        f"(All) #interactions: {len(df)}, #users: {n_users}, #items: {n_items}, #features_pos: {n_features_pos}, #features_neg: {n_features_neg}"
    )

    ratings = list(df["rating"])
    _rating_dist = Counter(ratings)
    rating_dist = {k: _rating_dist[k] for k in sorted(_rating_dist)}
    logger.info(f"rating count distribution: {rating_dist}")

    mean_exp_length = sum([len(tmp[2].split()) for tmp in list(df["template"])]) / len(
        df
    )
    logger.info(f"average explanation length: {mean_exp_length}")

    items = list(df["item"])
    _itemid_counts = Counter(items)
    average_itemid_frequency = (
        sum([count for count in _itemid_counts.values()]) / n_items
    )
    users = list(df["user"])
    _userid_counts = Counter(users)
    average_userid_frequency = (
        sum([count for count in _userid_counts.values()]) / n_users
    )
    logger.info(
        f"Average itemid frequency: {average_itemid_frequency} | Average userid frequency: {average_userid_frequency}"
    )

    for idx, role in zip([0, 1, 2], ["train", "valid", "test"]):
        _df = df[df["role"] == idx]
        n_features_pos = len(set([tmp[0] for tmp in list(_df["template"])]))
        n_features_neg = len(set([tmp[1] for tmp in list(_df["template"])]))
        logger.info(
            f"({role}) #interactions: {len(_df)}, #users: {len(set(_df['user']))}, #items: {len(set(_df['item']))}, #features_pos: {n_features_pos}, #features_neg: {n_features_neg}"
        )
        del _df

    reviews = df.to_dict(orient="records")
    return reviews
