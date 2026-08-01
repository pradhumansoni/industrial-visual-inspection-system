"""
Utilities for experiment reproducibility.

This module provides the SeedManager class, which ensures
reproducible behaviour across Python, NumPy and PyTorch.
"""

import random

import numpy as np
import torch


class SeedManager:
    """
    Utility class for setting random seeds across the project.
    """

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seeds for reproducible experiments.

        Parameters
        ----------
        seed : int
            Seed value to use.
        """

        # Python
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch (CPU)
        torch.manual_seed(seed)

        # PyTorch (GPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    SeedManager.set_seed(42)

    print(random.randint(1, 100))
    print(np.random.randint(1, 100))
    print(torch.randint(1, 100, (1,)))