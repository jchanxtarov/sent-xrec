"""Utility functions for model training and initialization.

This module provides helper functions for initializing and configuring the training
environment, data loaders, models, and trainers for recommendation systems. It includes
functionality for setting up logging, loading configurations, and managing model
training workflows.

Main components:
- Environment initialization
- Data loader configuration
- Model initialization
- Training setup
- Recommender pre-training
"""

import datetime
import os
import warnings
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2TokenizerFast, PreTrainedTokenizer

from loaders.common import XRecDataModule
from loaders.helpers import ReviewDataLoader
from loaders.loader import data_loader
from models.cer import CER
from models.common import Recommender
from models.erra import ERRA
from models.pepler import PEPLER
from models.pepler_d import PEPLER_D
from models.peter import PETER
from utils.tools import (
    create_logger,
    ensure_file,
    load_config,
    set_path,
    set_random_seed,
)

NOW = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def initialize() -> Tuple[DictConfig, Logger, str]:
    """Initialize training environment and configuration.

    Sets up CUDA, random seeds, logging, and loads configuration.

    Returns:
        Tuple containing:
            - config: DictConfig with model and training parameters
            - logger: Configured logging instance
            - save_root: Path to save model outputs
    """
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")

    config = load_config()
    set_random_seed(config.seed)

    log_name, save_root = set_path(
        model=config.model.name,
        dataset=config.dataset,
        uid=NOW,
        message=config.message,
    )
    if config.save_log:
        os.makedirs(save_root, exist_ok=True)

    logger = create_logger(
        name=__name__,
        file_name=os.path.join(save_root, f"{log_name}.log"),
        save=config.save_log,
    )
    logger.info("save_root = %s", save_root)
    logger.info("%s", OmegaConf.to_yaml(config))

    return config, logger, save_root


def get_loader(
    config: DictConfig,
    logger: Logger,
    for_recommender: bool = False,
) -> Tuple[LightningDataModule, Dict[str, Any], Optional[PreTrainedTokenizer]]:
    """Create data loader and tokenizer based on configuration.

    Args:
        config: Configuration dictionary containing model and training parameters
        logger: Logger instance for tracking setup progress
        for_recommender: Whether loader is for recommender pre-training

    Returns:
        Tuple containing:
            - loader: Configured PyTorch Lightning data module
            - stats: Dictionary of dataset statistics including counts and indices
            - tokenizer: Tokenizer instance if needed for the model, else None

    Raises:
        NotImplementedError: If specified dataset type is not supported
    """
    if config.dataset in ["ratebeer", "amazon_movie", "tripadvisor", "yelp"]:
        reviews = data_loader(
            logger,
            config.dataset,
            # config.max_seq_len, # Not used
            config.dev.max_data_size,
        )
    else:
        raise NotImplementedError()

    bos_token, eos_token, pad_token = "<bos>", "<eos>", "<pad>"
    tokenizer = None

    if config.model.name in ["pepler", "pepler_d"]:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            config.model.pretrained_model_name,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    loader = XRecDataModule(
        reviews=reviews,
        batch_size=(
            config.trainer.batch_size
            if not for_recommender
            else config.trainer_pretrain.batch_size
        ),
        max_seq_len=config.max_seq_len,
        max_vocab_size=config.max_vocab_size,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        tokenizer=tokenizer,
        is_recommender=for_recommender,
    )

    # Additional setup for specific models
    if config.model.name == "erra":
        logger.info("erra: add_aspects")
        loader.storage.add_aspects(config.model.retrieval_encoder_name)
        logger.info("erra: add_profiles")
        loader.storage.add_profiles(config.model.retrieval_encoder_name)
        logger.info("erra: finish retrieve and add aspects & profiles")

    if config.model.name == "pepler_d":
        logger.info("pepler_d: add_features_ui")
        loader.storage.add_features_ui(
            config.model.n_features, config.model.mode_retrieval
        )

    pad_idx = None
    if config.model.name not in ["pepler", "pepler_d"]:
        pad_idx = loader.storage.word_dict.word2idx["<pad>"]

    stats = {
        "n_users": len(loader.storage.user_dict),
        "n_items": len(loader.storage.item_dict),
        "n_tokens": len(loader.storage.word_dict),
        "max_rating": loader.storage.max_rating,
        "min_rating": loader.storage.min_rating,
        "pad_idx": pad_idx,
    }
    logger.info(stats)

    return loader, stats, tokenizer


def get_model(
    config: DictConfig,
    logger: Logger,
    stats: Dict[str, int],
    storage: ReviewDataLoader,
    save_root: str,
    tokenizer: Optional[PreTrainedTokenizer],
) -> Tuple[Optional[LightningModule], LightningModule]:
    """Create model instances based on configuration.

    Args:
        config: Configuration dictionary
        logger: Logger instance for tracking setup progress
        stats: Dataset statistics
        storage: Data storage instance
        save_root: Path to save outputs
        tokenizer: Tokenizer instance

    Returns:
        Tuple containing:
            - recommender: Optional recommender model
            - model: Main model instance

    Raises:
        NotImplementedError: If model type is not supported
    """
    # If logs aren't saved, set save_root to empty for safe references
    if not config.save_log:
        save_root = ""

    recommender = None
    if (
        hasattr(config.model, "type_rating_embedding")
        and config.model.type_rating_embedding is not None
    ):
        recommender = Recommender(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            storage=storage,
            pretrained_embed_name=(
                config.model.pretrained_model_name
                if hasattr(config.model, "pretrained_model_name")
                else None
            ),
            rec_type=config.pretrain.type,
            n_hidden_layers=config.pretrain.mlp_n_hidden_layers,
            d_hidden=config.pretrain.mlp_d_hidden,
            opt_lr=config.opt_pretrain.lr,
            opt_wd=config.opt_pretrain.wd,
            opt_factor=config.opt_pretrain.factor,
            opt_step_size=config.opt_pretrain.step_size,
            save_root=save_root,
        )

    # Main model initialization
    if config.model.name == "peter":
        src_len = 2
        if config.model.type_rating_embedding is not None:
            src_len += 1
        if config.model.use_feature:
            src_len += 1
        # NOTE: official implementation set #features as one.
        # see also: https://github.com/lileipisces/PETER/blob/master/utils.py#L204
        model = PETER(
            d_embed=config.model.d_embed,
            n_head=config.model.n_head,
            n_hid=config.model.n_hid,
            peter_mask=config.model.peter_mask,
            n_layers=config.model.n_layers,
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
            n_tokens=stats["n_tokens"],
            pad_idx=stats["pad_idx"],
            src_len=src_len,
            storage=storage,
            use_feature=config.model.use_feature,
            type_rating_embedding=config.model.type_rating_embedding,
            max_seq_len=config.max_seq_len,
            reg_text=config.model.reg_text,
            reg_context=config.model.reg_context,
            reg_rating=config.model.reg_rating,
            dropout=config.model.dropout,
            opt_lr=config.opt.lr,
            opt_wd=config.opt.wd,
            opt_factor=config.opt.factor,
            opt_step_size=config.opt.step_size,
            check_gen_text_every_n_epoch=config.trainer.check_gen_text_every_n_epoch,
            check_n_samples=config.trainer.check_n_samples,
            save_root=save_root,
        )
    elif config.model.name == "cer":
        model = CER(
            d_embed=config.model.d_embed,
            n_head=config.model.n_head,
            n_hid=config.model.n_hid,
            peter_mask=config.model.peter_mask,
            n_layers=config.model.n_layers,
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            src_len=3 if config.model.use_feature else 2,
            n_tokens=stats["n_tokens"],
            pad_idx=stats["pad_idx"],
            storage=storage,
            use_feature=config.model.use_feature,
            max_seq_len=config.max_seq_len,
            reg_text=config.model.reg_text,
            reg_context=config.model.reg_context,
            reg_rating=config.model.reg_rating,
            dropout=config.model.dropout,
            opt_lr=config.opt.lr,
            opt_wd=config.opt.wd,
            opt_factor=config.opt.factor,
            opt_step_size=config.opt.step_size,
            check_gen_text_every_n_epoch=config.trainer.check_gen_text_every_n_epoch,
            check_n_samples=config.trainer.check_n_samples,
            save_root=save_root,
        )
    elif config.model.name == "erra":
        # TODO: load from data
        # user_profile_embeds=torch.load('..//user_profile_embeds.pt')
        # item_profile_embeds=torch.load('..//item_profile_embeds.pt')
        model = ERRA(
            d_embed=config.model.d_embed,
            n_head=config.model.n_head,
            n_hid=config.model.n_hid,
            peter_mask=config.model.peter_mask,
            n_layers=config.model.n_layers,
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            src_len=6,  # [u, i, a1, a2, su, si]
            n_tokens=stats["n_tokens"],
            pad_idx=stats["pad_idx"],
            storage=storage,
            user_profile_embeds=storage.user_profile_embeds,
            item_profile_embeds=storage.item_profile_embeds,
            max_seq_len=config.max_seq_len,
            reg_text=config.model.reg_text,
            reg_context=config.model.reg_context,
            reg_rating=config.model.reg_rating,
            reg_aspect=config.model.reg_aspect,
            dropout=config.model.dropout,
            opt_lr=config.opt.lr,
            opt_wd=config.opt.wd,
            opt_factor=config.opt.factor,
            opt_step_size=config.opt.step_size,
            check_gen_text_every_n_epoch=config.trainer.check_gen_text_every_n_epoch,
            check_n_samples=config.trainer.check_n_samples,
            save_root=save_root,
        )
    elif config.model.name == "pepler":
        model = PEPLER(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
            tokenizer=tokenizer,
            storage=storage,
            type_rating_embedding=config.model.type_rating_embedding,
            use_seq_optimizers=config.trainer.use_seq_optimizers,
            max_seq_len=config.max_seq_len,
            reg_text=config.model.reg_text,
            reg_rating=config.model.reg_rating,
            pretrained_model_name=config.model.pretrained_model_name,
            rec_type=config.model.rec_model.type,
            n_hidden_layers=config.model.rec_model.mlp_n_hidden_layers,
            d_hidden=config.model.rec_model.mlp_d_hidden,
            opt_lr=config.opt.lr,
            opt_wd=config.opt.wd,
            opt_factor=config.opt.factor,
            opt_step_size=config.opt.step_size,
            check_gen_text_every_n_epoch=config.trainer.check_gen_text_every_n_epoch,
            check_n_samples=config.trainer.check_n_samples,
            save_root=save_root,
        )
    elif config.model.name == "pepler_d":
        model = PEPLER_D(
            tokenizer=tokenizer,
            storage=storage,
            max_seq_len=config.max_seq_len,
            pretrained_model_name=config.model.pretrained_model_name,
            n_keywords=config.model.n_features,
            opt_lr=config.opt.lr,
            opt_wd=config.opt.wd,
            opt_factor=config.opt.factor,
            opt_step_size=config.opt.step_size,
            check_gen_text_every_n_epoch=config.trainer.check_gen_text_every_n_epoch,
            check_n_samples=config.trainer.check_n_samples,
            save_root=save_root,
        )
    else:
        raise NotImplementedError()

    if (
        config.pretrain.ckpt_path != ""
        and config.model.type_rating_embedding is not None
        and recommender is not None
    ):
        ckpt = torch.load(config.pretrain.ckpt_path)
        recommender.load_state_dict(ckpt["state_dict"])
        logger.info("pretrained recommender loaded")

    if config.test.mode and config.test.ckpt_path != "":
        ckpt = torch.load(config.test.ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("pretrained model loaded")

    return recommender, model


def get_trainer(
    config: DictConfig,
    stats: Dict[str, Any],
    save_root: str,
    for_recommender: bool = False,
) -> Trainer:
    """Configure PyTorch Lightning trainer with callbacks and logging.

    Args:
        config: Configuration dictionary containing training parameters
        stats: Dictionary of dataset statistics
        save_root: Directory path for saving model outputs
        for_recommender: Whether trainer is for recommender pre-training

    Returns:
        Configured PyTorch Lightning Trainer instance with appropriate callbacks
        and settings based on the configuration
    """
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    callbacks = [progress_bar]

    monitor = "valid/loss"
    early_stop_callback = EarlyStopping(
        monitor=monitor if for_recommender is False else "pretrain/valid/loss",
        patience=(
            config.trainer.patience
            if for_recommender is False
            else config.trainer_pretrain.patience
        ),
        mode="min",
    )
    callbacks.append(early_stop_callback)

    if config.save_model:
        ensure_file(save_root)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor if for_recommender is False else "pretrain/valid/loss",
            mode="min",
            filename="best-{epoch}" if for_recommender is False else "recommender",
            dirpath=save_root,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)

    # Use W&B logger if logging is enabled
    logger = False
    if config.save_log:
        logger = WandbLogger(
            name=str(save_root)[-19:] + "_" + config.model.name,
            project=config.wandb_project,
            log_model=config.save_model,
            save_dir=save_root,
        )
        config_dict = OmegaConf.to_container(config, resolve=True)
        logger.experiment.config.update(config_dict)
        logger.experiment.config.update(stats)

    # Trainer config adjustments for pre-training vs. main training
    if for_recommender:
        trainer = Trainer(
            max_epochs=config.trainer_pretrain.epochs,
            logger=logger,
            callbacks=callbacks,
            devices=(
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else "auto"
            ),
            limit_train_batches=config.dev.limit_train_batches,  # None = full batches
            limit_val_batches=config.dev.limit_val_batches,  # None = full batches
            limit_test_batches=config.dev.limit_test_batches,  # None = full batches
        )
    else:
        trainer = Trainer(
            max_epochs=config.trainer.epochs,
            check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
            logger=logger,
            callbacks=callbacks,
            devices=(
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else "auto"
            ),
            limit_train_batches=config.dev.limit_train_batches,  # None = full batches
            limit_val_batches=config.dev.limit_val_batches,  # None = full batches
            limit_test_batches=config.dev.limit_test_batches,  # None = full batches
            gradient_clip_val=config.opt.clip_gradients,
        )

    return trainer


def pretrain_recommender(
    config: DictConfig,
    logger: Logger,
    trainer: Trainer,
    recommender: Recommender,
    dataloader: LightningDataModule,
    stats: Dict[str, Any],
    model: LightningModule,
) -> Tuple[LightningModule, LightningDataModule]:
    """Pretrain the recommender model and prepare data for main training.

    Handles the pretraining workflow for the recommender model, including:
    - Model training if no checkpoint is provided
    - Generating and storing predictions
    - Updating dataloader configuration for main training

    Args:
        config: Configuration dictionary
        logger: Logger instance for tracking progress
        trainer: PyTorch Lightning trainer instance
        recommender: Recommender model to be pretrained
        dataloader: Data loader module
        stats: Dataset statistics
        model: Main model instance

    Returns:
        Tuple containing:
            - model: Updated main model instance
            - dataloader: Updated data loader configured for main training
    """
    if config.model.type_rating_embedding is not None:
        train_predictions, valid_predictions, test_predictions = (
            None,
            None,
            None,
        )

        if config.ablation.leak_rating:
            logger.info("skip 1st stage prediction for leaking.")
        else:
            # Only fit if no checkpoint is provided
            if config.pretrain.ckpt_path == "":
                trainer.fit(recommender, datamodule=dataloader)
                logger.info("finish 1st stage pre-training.")

            # Evaluate and generate predictions
            _ = trainer.test(recommender, datamodule=dataloader)
            predict_dataloaders = dataloader.predict_dataloaders()

            _, train_predictions = get_predictions(
                recommender, predict_dataloaders["train"]
            )
            _, valid_predictions = get_predictions(
                recommender, predict_dataloaders["valid"]
            )
            test_predictions = recommender.outputs_test_step["rating_predict"]
            logger.info("finish 1st stage prediction.")

        if config.model.type_rating_embedding is not None:
            mae, rmse = dataloader.storage.add_pred_rating(
                train_ratings=train_predictions,
                valid_ratings=valid_predictions,
                test_ratings=test_predictions,
                leak_rating=config.ablation.leak_rating,
                noise_std=config.ablation.noise_std,
                min_rating=stats["min_rating"],
                max_rating=stats["max_rating"],
            )
            logger.info("finish registering rating_predict.")
            logger.info(f"noise | MAE: {mae} | RMSE: {rmse}")

    # Update data loader to main training settings
    dataloader.batch_size = config.trainer.batch_size
    logger.info("finish updating batch size.")

    dataloader.is_recommender = False
    logger.info("finish updating status (is_recommender).")

    return model, dataloader


def get_predictions(
    recommender: Recommender,
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[List[float], List[float]]:
    """Get rating predictions from a trained recommender model.

    Args:
        recommender: Trained recommender model instance
        dataloader: DataLoader containing user-item pairs for prediction

    Returns:
        Tuple containing:
            - targets: List of actual target ratings
            - predictions: List of model-predicted ratings
    """
    recommender.eval()
    recommender.freeze()

    targets = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            user, item, rating = batch
            pred_rating = recommender(user, item)
            targets.append(rating)

            # Ensure consistent shape
            if pred_rating.dim() == 0:
                pred_rating = pred_rating.unsqueeze(0)
            predictions.append(pred_rating)

    return (
        torch.cat(targets, dim=0).tolist(),
        torch.cat(predictions, dim=0).tolist(),
    )
