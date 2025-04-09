"""
Statistical operations on PyTorch tensors.

This module provides functions for calculating various statistics on tensors.
"""

import torch


def tensor_statistics(tensor):
    """
    Calculate various statistics for a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        dict: Dictionary containing various statistics
    """
    stats = {
        'mean': torch.mean(tensor.double()).item(),
        'std': torch.std(tensor.double()).item(),
        'min': torch.min(tensor).item(),
        'max': torch.max(tensor).item(),
        'sum': torch.sum(tensor.double()).item()
    }
    return stats


def row_wise_sum(tensor):
    """
    Calculate the sum of each row in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: A tensor containing row-wise sums
    """
    return torch.sum(tensor, dim=-1)


def column_wise_sum(tensor):
    """
    Calculate the sum of each column in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: A tensor containing column-wise sums
    """
    return torch.sum(tensor, dim=-2)


def image_batch_statistics(images):
    """
    Calculate statistics for a batch of images.

    Args:
        images (torch.Tensor): Batch of images with shape (batch_size, channels, height, width)

    Returns:
        dict: Dictionary containing statistics for each image
    """
    batch_size = images.shape[0]
    stats = []

    for i in range(batch_size):
        image_stats = {
            'image_id': i,
            'row_sums': row_wise_sum(images[i]),
            'column_sums': column_wise_sum(images[i]),
            'mean': torch.mean(images[i].double()).item(),
            'std': torch.std(images[i].double()).item()
        }
        stats.append(image_stats)

    # Also compute statistics across the entire batch
    all_stats = {
        'batch_mean': torch.mean(images.double()).item(),
        'batch_std': torch.std(images.double()).item(),
        'batch_min': torch.min(images).item(),
        'batch_max': torch.max(images).item()
    }

    return stats, all_stats


def find_mode(tensor):
    """
    Find the most frequent value in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        tuple: (mode_value, frequency)
    """
    # Flatten the tensor to process all values
    flattened = tensor.reshape(-1)

    # Get unique values and their counts
    unique_values, counts = torch.unique(flattened, return_counts=True)

    # Find the index of the maximum count
    max_idx = torch.argmax(counts)

    # Return the mode and its frequency
    return unique_values[max_idx].item(), counts[max_idx].item()
