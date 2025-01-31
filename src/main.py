from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as pl
from transformers import PreTrainedTokenizer

from utils.helpers import (
    get_loader,
    get_model,
    get_trainer,
    initialize,
    pretrain_recommender,
)


def main() -> None:
    """Main function to run the recommendation system training pipeline.

    This function orchestrates the entire training process:
    1. Initializes configuration, logger, and directories
    2. Sets up the data loader with appropriate configuration
    3. Initializes the model(s)
    4. Handles pretraining for recommender models if needed
    5. Runs the main training loop
    6. Evaluates the model on test data
    """
    config, logger, save_root = initialize()

    # Determine if we're working with a recommender system
    is_recommender = False
    if (
        hasattr(config.model, "type_rating_embedding")
        and config.model.type_rating_embedding is not None
    ):
        is_recommender = True

    # Get data loader and associated components
    dataloader, stats, tokenizer = get_loader(config, logger, is_recommender)

    # Initialize models
    recommender, model = get_model(
        config,
        stats,
        dataloader.storage,
        save_root,
        tokenizer,
    )

    # Handle pretraining for specific recommender models
    if (
        hasattr(config.model, "type_rating_embedding")
        and config.model.type_rating_embedding is not None
    ):
        if config.model.name in ["peter", "pepler"]:
            trainer = get_trainer(config, stats, save_root, True)
            model, dataloader = pretrain_recommender(
                config, logger, trainer, recommender, dataloader, stats, model
            )
        else:
            raise NotImplementedError("Unsupported recommender model type")

    # Set up trainer and run training/testing
    trainer = get_trainer(config, stats, save_root)
    if config.prediction_mode is not True:
        trainer.fit(model, datamodule=dataloader)

    # Run evaluation and log results
    scores = trainer.test(model, datamodule=dataloader)
    logger.info(scores)


if __name__ == "__main__":
    main()
