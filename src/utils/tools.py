"""Utility functions for file handling, configuration, and logging setup."""

import logging
import os
import random
from logging import config
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

FORMATER_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "conf/logger.yaml",
)


def set_path(
    model: str,
    dataset: str,
    uid: str,
    message: str,
) -> Tuple[str, str]:
    """Set up logging and save paths for model outputs.

    Args:
        model: Name of the model
        dataset: Name of the dataset
        uid: Unique identifier (usually timestamp)
        message: Additional message to append to paths

    Returns:
        Tuple containing:
            - log_name: Name for log file
            - save_root: Root directory for saving outputs
    """
    torch.cuda.empty_cache()

    log_name = f"{model}_{message}"
    dir = f"{uid}" if message == "" else f"{uid}_{model}_{message}"
    save_root = f"./outputs/{dataset}/{dir}"

    return log_name, save_root


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU and GPU operations.

    Args:
        seed: Integer seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(name: str = "common") -> DictConfig:
    """Load and merge configuration from files and command line.

    Args:
        name: Base configuration name to load

    Returns:
        Merged configuration object
    """
    cli_conf = OmegaConf.from_cli()
    base_conf = OmegaConf.load(f"conf/{name}.yaml")
    base_conf = OmegaConf.merge(base_conf, cli_conf)
    exp_conf = OmegaConf.load(cli_conf.config)
    merged_conf = OmegaConf.merge(exp_conf, base_conf)
    return merged_conf


def generate_path_save_root(dataset: str, model: str) -> str:
    """Generate root path for saving model outputs.

    Args:
        dataset: Name of the dataset
        model: Name of the model

    Returns:
        Path string for saving outputs
    """
    return f"./outputs/{dataset}/{model}"


def create_config() -> DictConfig:
    """Create configuration from command line arguments.

    Returns:
        Configuration object merged with CLI arguments
    """
    cli_conf = OmegaConf.from_cli()
    base_conf = OmegaConf.load(cli_conf.config)
    merged_conf = OmegaConf.merge(base_conf, cli_conf)
    return merged_conf


def create_logger(
    name: str, file_name: str, save: bool = True
) -> logging.Logger:
    """Create and configure logger instance.

    Args:
        name: Logger name
        file_name: Path to log file
        save: Whether to save logs to file

    Returns:
        Configured logger instance
    """
    with open(FORMATER_PATH, "r") as f:
        logger_config = yaml.safe_load(f)

    logger_config["handlers"]["file"]["filename"] = file_name
    if save is False:
        _ = logger_config["handlers"].pop("file")
        logger_config["loggers"]["utils.helpers"]["handlers"] = ["console"]
        logger_config["loggers"]["utils.helpers_eval"]["handlers"] = [
            "console"
        ]

    config.dictConfig(logger_config)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger


def ensure_file(path: str) -> None:
    """Ensure directory exists for given file path.

    Args:
        path: File path to check/create directory for
    """
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
