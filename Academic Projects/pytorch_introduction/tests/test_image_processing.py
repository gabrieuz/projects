"""
Tests for image processing module.

This module contains unit tests for the image processing functions.
"""

import unittest
import torch
from src.image_processing import (
    create_synthetic_image,
    create_image_batch,
    get_image_rows,
    get_image_columns,
    flatten_image,
)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing module."""

    def test_create_synthetic_image(self):
        """Test creating synthetic images with random pixel values."""
        height, width = 10, 15
        image = create_synthetic_image(height, width)

        # Test shape
        self.assertEqual(image.shape, (height, width))

        # Test value range
        self.assertTrue(torch.all(image >= 0).item())
        self.assertTrue(torch.all(image <= 255).item())

        # Test custom value range
        min_val, max_val = 10, 200
        image_custom = create_synthetic_image(
            height, width, min_val=min_val, max_val=max_val)
        self.assertTrue(torch.all(image_custom >= min_val).item())
        self.assertTrue(torch.all(image_custom <= max_val).item())

        # Test seed reproducibility
        seed = 42
        image1 = create_synthetic_image(height, width, seed=seed)
        image2 = create_synthetic_image(height, width, seed=seed)
        self.assertTrue(torch.equal(image1, image2))

    def test_create_image_batch(self):
        """Test creating batches of synthetic images."""
        batch_size = 5
        height, width = 8, 8
        channels = 3
        batch = create_image_batch(batch_size, height, width, channels)

        # Test batch shape
        self.assertEqual(batch.shape, (batch_size, channels, height, width))

        # Test single channel case
        single_channel_batch = create_image_batch(
            batch_size, height, width, channels=1)
        self.assertEqual(single_channel_batch.shape,
                         (batch_size, 1, height, width))

        # Test seed reproducibility
        seed = 42
        batch1 = create_image_batch(
            batch_size, height, width, channels, seed=seed)
        batch2 = create_image_batch(
            batch_size, height, width, channels, seed=seed)
        self.assertTrue(torch.equal(batch1, batch2))

    def test_get_image_rows(self):
        """Test extracting even or odd rows from an image."""
        # Create a test image with sequential values for easy testing
        image = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])

        # Test odd rows (default)
        odd_rows = get_image_rows(image, even_rows=False)
        expected_odd = torch.tensor([
            [1, 2, 3],
            [7, 8, 9]
        ])
        self.assertTrue(torch.equal(odd_rows, expected_odd))

        # Test even rows
        even_rows = get_image_rows(image, even_rows=True)
        expected_even = torch.tensor([
            [4, 5, 6],
            [10, 11, 12]
        ])
        self.assertTrue(torch.equal(even_rows, expected_even))

        # Test with batch dimensions
        # Create a batch of 2 identical images
        batch_image = image.unsqueeze(0).repeat(2, 1, 1)
        batch_odd_rows = get_image_rows(batch_image)
        self.assertEqual(batch_odd_rows.shape, (2, 2, 3))

    def test_get_image_columns(self):
        """Test extracting even or odd columns from an image."""
        # Create a test image with sequential values for easy testing
        image = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])

        # Test odd columns (default)
        odd_cols = get_image_columns(image, even_columns=False)
        expected_odd = torch.tensor([
            [1, 3],
            [5, 7],
            [9, 11]
        ])
        self.assertTrue(torch.equal(odd_cols, expected_odd))

        # Test even columns
        even_cols = get_image_columns(image, even_columns=True)
        expected_even = torch.tensor([
            [2, 4],
            [6, 8],
            [10, 12]
        ])
        self.assertTrue(torch.equal(even_cols, expected_even))

        # Test with batch dimensions
        batch_image = image.unsqueeze(0).unsqueeze(
            0)  # Add batch and channel dimensions
        batch_odd_cols = get_image_columns(batch_image)
        self.assertEqual(batch_odd_cols.shape, (1, 1, 3, 2))

    def test_flatten_image(self):
        """Test flattening spatial dimensions of an image tensor."""
        batch_size = 2
        channels = 3
        height, width = 4, 5

        # Create test batch
        image_batch = torch.randn(batch_size, channels, height, width)

        # Test flattening
        flattened = flatten_image(image_batch)
        self.assertEqual(
            flattened.shape, (batch_size, channels, height * width))

        # Test values preserved after flattening
        original_first_pixel = image_batch[0, 0, 0, 0].item()
        flattened_first_pixel = flattened[0, 0, 0].item()
        self.assertEqual(original_first_pixel, flattened_first_pixel)

        # Test that the flattened tensor contains all the same values as the original
        self.assertTrue(torch.equal(image_batch.reshape(
            batch_size, channels, -1), flattened))


if __name__ == "__main__":
    unittest.main()
