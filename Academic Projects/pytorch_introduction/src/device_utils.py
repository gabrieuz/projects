"""
Device utilities for PyTorch.

This module provides functions for checking GPU availability and setting up the device for PyTorch operations.
"""

import torch


def get_device():
    """
    Check if CUDA is available and return the appropriate device.

    Returns:
        torch.device: The device to use for tensor operations (CUDA if available, otherwise CPU)
        str: A message indicating which device is being used
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device('cuda')
        message = f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}"
    else:
        device = torch.device("cpu")
        message = "CUDA not available. Using CPU..."

    return device, message


def set_seed(seed=None):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int, optional): The seed value. If None, use the default seed.

    Returns:
        int: The seed value used
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.initial_seed()


def print_device_info():
    """
    Print information about the device being used.

    Returns:
        dict: A dictionary containing device information
    """
    device, message = get_device()
    seed = torch.initial_seed()

    print(message)
    print(f"Random seed: {seed}")

    return {
        "device": device,
        "seed": seed,
        "cuda_available": torch.cuda.is_available()
    }
