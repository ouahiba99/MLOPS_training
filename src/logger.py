"""
Central logging setup. Every module gets its logger from here instead of
using print() — this is what item 3 of Task 3 asks for: log levels, log
format, log to file and to console.
"""

import logging
import sys

from src.config import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers if get_logger is called twice
        return logger

    logger.setLevel(config.logging.level)
    formatter = logging.Formatter(config.logging.format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    config.logging.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.logging.log_dir / "service.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
