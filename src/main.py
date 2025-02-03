"""
Main module to run the recommendation system training pipeline.
"""

from utils.helpers import (
    get_loader,
    get_model,
    get_trainer,
    initialize,
    pretrain_recommender,
)


def main() -> None:
    """
    Main function to run the recommendation system training pipeline.

    This function orchestrates the entire training process:
      1. Initializes configuration, logger, and directories
      2. Sets up the data loader with appropriate configuration
      3. Initializes the model(s)
      4. Handles pre-training for recommender models if needed
      5. Runs the main training loop
      6. Evaluates the model on test data
    """

    # Initialize configuration, logger, and root directory for saving
    config, logger, save_root = initialize()

    # Determine if we're working with a recommender system
    is_recommender = bool(
        hasattr(config.model, "type_rating_embedding")
        and config.model.type_rating_embedding is not None
    )

    # Get data loader and associated components based on the config
    dataloader, stats, tokenizer = get_loader(config, logger, is_recommender)

    # Initialize models
    recommender, model = get_model(
        config,
        logger,
        stats,
        dataloader.storage,
        save_root,
        tokenizer,
    )

    # If a rating-embedding-based recommender model is specified, handle pre-training
    if is_recommender:
        if config.model.name in ["peter", "pepler"]:
            # Create a trainer with pre-training configuration
            trainer = get_trainer(config, stats, save_root, for_recommender=True)
            # Pretrain the recommender model before the main training loop
            model, dataloader = pretrain_recommender(
                config, logger, trainer, recommender, dataloader, stats, model
            )
        else:
            raise NotImplementedError("Unsupported recommender model type")

    # Set up trainer (for main training and testing)
    trainer = get_trainer(config, stats, save_root)

    # If we are not in prediction mode, fit the model
    if not config.test.mode:
        trainer.fit(model, datamodule=dataloader)

    # Test the trained model and log results
    scores = trainer.test(model, datamodule=dataloader)
    logger.info(scores)


if __name__ == "__main__":
    main()
