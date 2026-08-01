"""
Configuration management utilities.

This module provides:

1. Config
   - Converts nested dictionaries into Python objects.
   - Enables attribute-style access.

2. ConfigLoader
   - Loads YAML configuration files.
   - Returns a Config object.
"""

from pathlib import Path

import yaml


class Config:
    """
    Converts nested dictionaries into Python objects,
    enabling attribute-style access.
    """

    def __init__(self, data: dict):

        for key, value in data.items():

            if isinstance(value, dict):
                value = Config(value)

            setattr(self, key, value)


class ConfigLoader:
    """
    Loads YAML configuration files
    and converts them into Config objects.
    """

    def __init__(self, config_path: str | Path):

        self.config_path = Path(config_path)

    def load(self) -> Config:
        """
        Load configuration file.

        Returns
        -------
        Config
            Parsed configuration object.
        """

        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return Config(data)


if __name__ == "__main__":

    config = ConfigLoader("../../configs/training.yaml").load()

    print(config.training.batch_size)
    print(config.training.epochs)

