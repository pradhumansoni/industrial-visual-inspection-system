"""
Device management utilities.

This module provides a single interface for selecting the
best available compute device.
"""

import torch


class DeviceManager:
    """
    Utility class for managing compute devices.
    """

    @staticmethod
    def get_device() -> torch.device:
        """
        Return the best available compute device.

        Priority
        --------
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU
        """

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def device_name() -> str:
        """
        Return a human-readable device name.
        """

        device = DeviceManager.get_device()

        if device.type == "cuda":
            return torch.cuda.get_device_name(0)

        if device.type == "mps":
            return "Apple Silicon GPU"

        return "CPU"

    @staticmethod
    def print_device_info() -> None:
        """
        Print information about the selected device.
        """

        device = DeviceManager.get_device()

        print("=" * 50)
        print("Device Information")
        print("=" * 50)
        print(f"Device      : {device}")
        print(f"Device Name : {DeviceManager.device_name()}")

        if device.type == "cuda":
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count   : {torch.cuda.device_count()}")

        print("=" * 50)


if __name__ == "__main__":

    DeviceManager.print_device_info()