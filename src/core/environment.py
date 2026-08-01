"""
Environment management utilities.

This module collects environment metadata for reproducible
machine learning experiments.
"""

from __future__ import annotations

import platform
import sys

import torch

from src.core.device import DeviceManager


class EnvironmentManager:
    """
    Collects information about the execution environment.
    """

    DEFAULT_SEED = 42

    @staticmethod
    def get_environment(seed: int | None = None) -> dict:
        """
        Collect environment information.

        Parameters
        ----------
        seed : int | None
            Experiment seed.

        Returns
        -------
        dict
            Dictionary containing environment metadata.
        """

        device = DeviceManager.get_device()

        environment = {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": device.type,
            "seed": seed if seed is not None else EnvironmentManager.DEFAULT_SEED,
            "operating_system": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

        if device.type == "cuda":

            environment.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_mb": round(
                        torch.cuda.get_device_properties(0).total_memory
                        / (1024 ** 2)
                    ),
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                }
            )

        else:

            environment.update(
                {
                    "gpu": None,
                    "gpu_count": 0,
                    "gpu_memory_mb": None,
                    "cuda": None,
                    "cudnn": None,
                }
            )

        return environment

    @staticmethod
    def print_environment(seed: int | None = None) -> None:
        """
        Print collected environment information.
        """

        environment = EnvironmentManager.get_environment(seed)

        print("=" * 60)
        print("Environment Information")
        print("=" * 60)

        for key, value in environment.items():
            print(f"{key:20}: {value}")

        print("=" * 60)


if __name__ == "__main__":

    EnvironmentManager.print_environment()