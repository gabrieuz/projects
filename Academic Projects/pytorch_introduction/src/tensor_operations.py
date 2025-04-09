"""
Basic tensor operations for PyTorch.

This module provides functions for creating and manipulating PyTorch tensors.
"""

import torch


def create_random_tensor(shape, seed=None):
    """
    Create a random tensor with the specified shape.

    Args:
        shape (tuple): Shape of the tensor to create
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: A random tensor with values between 0 and 1
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.rand(shape)


def matrix_multiplication(tensor1, tensor2):
    """
    Perform matrix multiplication between two tensors.

    Args:
        tensor1 (torch.Tensor): First tensor
        tensor2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: Result of the matrix multiplication
    """
    return torch.matmul(tensor1, tensor2)


def element_wise_multiplication(tensor1, tensor2):
    """
    Perform element-wise multiplication between two tensors.

    Args:
        tensor1 (torch.Tensor): First tensor
        tensor2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: Result of the element-wise multiplication
    """
    return torch.mul(tensor1, tensor2)


def find_min_max(tensor):
    """
    Find the minimum and maximum values in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        tuple: (min_value, max_value, min_index, max_index)
    """
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    min_index = torch.argmin(tensor)
    max_index = torch.argmax(tensor)

    return min_value.item(), max_value.item(), min_index.item(), max_index.item()


def squeeze_tensor(tensor):
    """
    Remove all dimensions of size 1 from a tensor.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Tensor with all singleton dimensions removed
    """
    return tensor.squeeze()


def unsqueeze_tensor(tensor, dim=0):
    """
    Add a dimension of size 1 to a tensor at the specified position.

    Args:
        tensor (torch.Tensor): Input tensor
        dim (int): Position to add the new dimension

    Returns:
        torch.Tensor: Tensor with an additional dimension
    """
    return tensor.unsqueeze(dim)
