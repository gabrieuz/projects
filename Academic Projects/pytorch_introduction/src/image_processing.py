"""
Image processing utilities using PyTorch tensors.

This module provides functions for creating synthetic images and image manipulation using tensors.
"""

import torch


def create_synthetic_image(height, width, min_val=0, max_val=255, seed=None):
    """
    Create a synthetic image with random pixel values.

    Args:
        height (int): Height of the image in pixels
        width (int): Width of the image in pixels
        min_val (int): Minimum pixel value
        max_val (int): Maximum pixel value
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: A tensor representing an image with shape (height, width)
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randint(min_val, max_val + 1, (height, width))


def create_image_batch(batch_size, height, width, channels=1, min_val=0, max_val=255, seed=None):
    """
    Create a batch of synthetic images.

    Args:
        batch_size (int): Number of images to create
        height (int): Height of each image in pixels
        width (int): Width of each image in pixels
        channels (int): Number of channels per image
        min_val (int): Minimum pixel value
        max_val (int): Maximum pixel value
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: A tensor representing a batch of images with shape (batch_size, channels, height, width)
    """
    if seed is not None:
        torch.manual_seed(seed)

    images = torch.stack([create_synthetic_image(height, width, min_val, max_val)
                         for _ in range(batch_size)])

    # Add channel dimension if required
    if channels > 0:
        images = images.unsqueeze(1)

    return images


def get_image_rows(image, even_rows=False):
    """
    Extract even or odd rows from an image.

    Args:
        image (torch.Tensor): Input image tensor with shape (..., height, width)
        even_rows (bool): If True, return even-indexed rows, otherwise return odd-indexed rows

    Returns:
        torch.Tensor: Tensor containing the selected rows
    """
    start_idx = 1 if even_rows else 0
    return image[..., start_idx::2, :]


def get_image_columns(image, even_columns=False):
    """
    Extract even or odd columns from an image.

    Args:
        image (torch.Tensor): Input image tensor with shape (..., height, width)
        even_columns (bool): If True, return even-indexed columns, otherwise return odd-indexed columns

    Returns:
        torch.Tensor: Tensor containing the selected columns
    """
    start_idx = 1 if even_columns else 0
    return image[..., :, start_idx::2]


def flatten_image(image):
    """
    Flatten the spatial dimensions of an image tensor.

    Args:
        image (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width)

    Returns:
        torch.Tensor: Flattened tensor with shape (batch_size, channels, height * width)
    """
    batch_size = image.shape[0]
    channels = image.shape[1]
    return image.reshape(batch_size, channels, -1)
