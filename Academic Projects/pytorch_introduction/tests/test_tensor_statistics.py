"""
Tests for tensor statistics module.

This module contains unit tests for the tensor statistics functions.
"""

import unittest
import torch
import numpy as np
from src.tensor_statistics import (
    tensor_statistics,
    row_wise_sum,
    column_wise_sum,
    image_batch_statistics,
    find_mode,
)


class TestTensorStatistics(unittest.TestCase):
    """Test cases for tensor statistics module."""

    def test_tensor_statistics(self):
        """Test calculating various statistics for a tensor."""
        # Create a tensor with known statistics
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = tensor_statistics(tensor)

        # Check if all statistics are calculated correctly
        self.assertAlmostEqual(stats['mean'], 3.5)
        self.assertAlmostEqual(stats['std'], 1.7078251, places=6)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 6.0)
        self.assertEqual(stats['sum'], 21.0)

    def test_row_wise_sum(self):
        """Test calculating the sum of each row in a tensor."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = row_wise_sum(tensor)

        expected = torch.tensor([6.0, 15.0])
        self.assertTrue(torch.allclose(result, expected))

        # Test with 3D tensor
        tensor3d = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        result3d = row_wise_sum(tensor3d)
        expected3d = torch.tensor([[3.0, 7.0], [11.0, 15.0]])
        self.assertTrue(torch.allclose(result3d, expected3d))

    def test_column_wise_sum(self):
        """Test calculating the sum of each column in a tensor."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = column_wise_sum(tensor)

        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertTrue(torch.allclose(result, expected))

        # Test with 3D tensor
        tensor3d = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        result3d = column_wise_sum(tensor3d)
        expected3d = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
        self.assertTrue(torch.allclose(result3d, expected3d))

    def test_image_batch_statistics(self):
        """Test calculating statistics for a batch of images."""
        # Create a small batch of "images" (3 images, 2 channels, 2x2 pixels)
        batch = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],  # Image 1
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],  # Image 2
            [[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]  # Image 3
        ])

        image_stats, batch_stats = image_batch_statistics(batch)

        # Check number of image stats
        self.assertEqual(len(image_stats), 3)

        # Check image 0 stats
        self.assertEqual(image_stats[0]['image_id'], 0)
        self.assertAlmostEqual(image_stats[0]['mean'], 4.5)

        # Check batch stats
        self.assertAlmostEqual(batch_stats['batch_mean'], 12.5)
        self.assertEqual(batch_stats['batch_min'], 1.0)
        self.assertEqual(batch_stats['batch_max'], 24.0)

        # Check row and column sums for first image
        expected_row_sums = torch.tensor([[3.0, 7.0], [11.0, 15.0]])
        expected_col_sums = torch.tensor([[4.0, 6.0], [12.0, 14.0]])
        self.assertTrue(torch.allclose(
            image_stats[0]['row_sums'], expected_row_sums))
        self.assertTrue(torch.allclose(
            image_stats[0]['column_sums'], expected_col_sums))

    def test_find_mode(self):
        """Test finding the most frequent value in a tensor."""
        # Test with distinct values
        tensor = torch.tensor([1, 2, 3, 2, 2, 1, 2])
        mode_value, frequency = find_mode(tensor)
        self.assertEqual(mode_value, 2)
        self.assertEqual(frequency, 4)

        # Test with multiple dimensions
        tensor2d = torch.tensor([[1, 2, 1], [2, 3, 2]])
        mode_value2d, frequency2d = find_mode(tensor2d)
        self.assertEqual(mode_value2d, 2)
        self.assertEqual(frequency2d, 3)

        # Test with floating point values
        tensor_float = torch.tensor([1.1, 2.2, 1.1, 3.3, 1.1])
        mode_value_float, frequency_float = find_mode(tensor_float)
        self.assertEqual(mode_value_float, 1.1)
        self.assertEqual(frequency_float, 3)


if __name__ == "__main__":
    unittest.main()
