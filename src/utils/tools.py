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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "conf/logger.yaml",
)


def set_path(
    model: str,
    dataset: str,
    uid: str,
    message: str,
) -> Tuple[str, str]:
    torch.cuda.empty_cache()

    log_name = f"{model}_{message}"
    dir = f"{uid}" if message == "" else f"{uid}_{model}_{message}"
    save_root = f"./outputs/{dataset}/{dir}"

    return log_name, save_root


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(name: str = "common") -> Any:  # -> DictConfig
    cli_conf = OmegaConf.from_cli()
    base_conf = OmegaConf.load(f"conf/{name}.yaml")
    base_conf = OmegaConf.merge(base_conf, cli_conf)
    exp_conf = OmegaConf.load(cli_conf.config)
    merged_conf = OmegaConf.merge(exp_conf, base_conf)
    return merged_conf


def generate_path_save_root(dataset: str, model: str) -> str:
    return f"./outputs/{dataset}/{model}"


def create_config() -> Any:  # -> DictConfig
    cli_conf = OmegaConf.from_cli()
    base_conf = OmegaConf.load(cli_conf.config)
    merged_conf = OmegaConf.merge(base_conf, cli_conf)
    return merged_conf


def create_logger(name: str, file_name: str, save: bool = True) -> logging.Logger:
    with open(FORMATER_PATH, "r") as f:
        logger_config = yaml.safe_load(f)

    logger_config["handlers"]["file"]["filename"] = file_name
    if save is False:
        _ = logger_config["handlers"].pop("file")
        logger_config["loggers"]["utils.helpers"]["handlers"] = ["console"]
        logger_config["loggers"]["utils.helpers_eval"]["handlers"] = ["console"]

    config.dictConfig(logger_config)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger


def ensure_file(path: str) -> None:
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
