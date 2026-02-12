"""Structured logging configuration helpers."""

import logging.config
import yaml
from pathlib import Path


def setup_logging(log_config_path: str = "configs/logging.yaml") -> logging.Logger:
    """
    Setup logging configuration from YAML file.
    
    Args:
        log_config_path: Path to logging configuration YAML file
        
    Returns:
        Root logger instance
    """
    config_path = Path(__file__).parent.parent.parent / log_config_path
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    
    return logging.getLogger()
