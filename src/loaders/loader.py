import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd


def data_loader(
    logger: logging.Logger,
    dataset: str,
    max_data_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and preprocess review dataset.

    This function loads a review dataset from a pickle file, processes it by splitting into
    train/valid/test sets, and collects statistics about the dataset.

    Args:
        logger (logging.Logger): Logger instance for recording information
        dataset (str): Name of the dataset to load
        max_data_size (Optional[int], optional): Maximum number of reviews to load. Defaults to None

    Returns:
        List[Dict[str, Any]]: List of review records containing user, item, rating, and template information

    Note:
        The function assumes the dataset file is located in 'datasets/{dataset}_exps.pkl.gz'
        relative to the project root directory.
    """
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    file_path = os.path.join(base_path, f"datasets/{dataset}_exps.pkl.gz")

    df = pd.read_pickle(file_path, compression="gzip")

    if max_data_size is not None:
        df = df[:max_data_size]
        logger.warning(
            f"*** Development mode with max_data_size = {max_data_size} ***"
        )
        logger.info(
            f"(After deletion using max_data_size) #reviews: {len(df)}"
        )
        df["role"] = 0
        df["index"] = [i for i in range(len(df))]
        df.loc[df.groupby("user").index.idxmax(), "role"] = 2
        _df_except_test = df.drop(
            df.loc[df.groupby("user").index.idxmax()].index
        )
        df.loc[_df_except_test.groupby("user").index.idxmax(), "role"] = 1
        del _df_except_test

    # Calculate dataset statistics
    n_users = len(set(df["user"]))
    n_items = len(set(df["item"]))
    n_features_pos = len(set([tmp[0] for tmp in list(df["template"])]))
    n_features_neg = len(set([tmp[1] for tmp in list(df["template"])]))
    logger.info(
        f"(All) #interactions: {len(df)}, #users: {n_users}, #items: {n_items}, #features_pos: {n_features_pos}, #features_neg: {n_features_neg}"
    )

    # Log rating distribution
    ratings = list(df["rating"])
    _rating_dist = Counter(ratings)
    rating_dist = {k: _rating_dist[k] for k in sorted(_rating_dist)}
    logger.info(f"rating count distribution: {rating_dist}")

    # Calculate average explanation length
    mean_exp_length = sum(
        [len(tmp[2].split()) for tmp in list(df["template"])]
    ) / len(df)
    logger.info(f"average explanation length: {mean_exp_length}")

    # Calculate item and user frequency statistics
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

    # Log statistics for each data split
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
