"""Shared utilities for the training pipeline."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root (the folder containing Makefile)."""
    # Walk up from this file until we find the Makefile
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "Makefile").exists():
            return parent
    # Fallback: use current working directory
    return Path(os.getcwd())
