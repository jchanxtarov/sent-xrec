from utils.helpers import (
    get_loader,
    get_model,
    get_trainer,
    initialize,
    pretrain_recommender,
)


def main():
    config, logger, save_root = initialize()

    is_recommender = False
    if (
        hasattr(config.model, "type_rating_embedding")
        and config.model.type_rating_embedding is not None
    ):
        is_recommender = True
    dataloader, stats, tokenizer = get_loader(config, logger, is_recommender)

    recommender, model = get_model(
        config,
        stats,
        dataloader.storage,
        save_root,
        tokenizer,
    )

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
            raise NotImplementedError

    trainer = get_trainer(config, stats, save_root)
    if config.prediction_mode is not True:
        trainer.fit(model, datamodule=dataloader)
    scores = trainer.test(model, datamodule=dataloader)
    logger.info(scores)


if __name__ == "__main__":
    main()
