"""
Tests for tensor operations module.

This module contains unit tests for the tensor operations functions.
"""

import unittest
import torch
import numpy as np
from src.tensor_operations import (
    create_random_tensor,
    matrix_multiplication,
    element_wise_multiplication,
    find_min_max,
    squeeze_tensor,
    unsqueeze_tensor,
)


class TestTensorOperations(unittest.TestCase):
    """Test cases for tensor operations module."""

    def test_create_random_tensor(self):
        """Test creating random tensors."""
        # Test shape
        shape = (3, 4)
        tensor = create_random_tensor(shape)
        self.assertEqual(tensor.shape, shape)

        # Test values range
        self.assertTrue(torch.all(tensor >= 0).item())
        self.assertTrue(torch.all(tensor <= 1).item())

        # Test seed reproducibility
        seed = 42
        tensor1 = create_random_tensor(shape, seed)
        tensor2 = create_random_tensor(shape, seed)
        self.assertTrue(torch.equal(tensor1, tensor2))

    def test_matrix_multiplication(self):
        """Test matrix multiplication between two tensors."""
        tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_multiplication(tensor1, tensor2)

        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_element_wise_multiplication(self):
        """Test element-wise multiplication between two tensors."""
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        result = element_wise_multiplication(tensor1, tensor2)

        expected = torch.tensor([4.0, 10.0, 18.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_find_min_max(self):
        """Test finding min and max values in a tensor."""
        tensor = torch.tensor([[1.0, 8.0, 3.0], [4.0, 2.0, 6.0]])
        min_value, max_value, min_index, max_index = find_min_max(tensor)

        self.assertEqual(min_value, 1.0)
        self.assertEqual(max_value, 8.0)
        self.assertEqual(min_index, 0)  # Flat index of the minimum value
        self.assertEqual(max_index, 1)  # Flat index of the maximum value

    def test_squeeze_tensor(self):
        """Test removing singleton dimensions from a tensor."""
        tensor = torch.rand(1, 3, 1, 4)
        result = squeeze_tensor(tensor)

        self.assertEqual(result.shape, (3, 4))

    def test_unsqueeze_tensor(self):
        """Test adding a singleton dimension to a tensor."""
        tensor = torch.rand(3, 4)

        # Test adding dimension at the beginning
        result = unsqueeze_tensor(tensor)
        self.assertEqual(result.shape, (1, 3, 4))

        # Test adding dimension in the middle
        result = unsqueeze_tensor(tensor, dim=1)
        self.assertEqual(result.shape, (3, 1, 4))


if __name__ == "__main__":
    unittest.main()
