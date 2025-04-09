"""
Tests for device utilities module.

This module contains unit tests for the device utilities functions.
"""

import unittest
import torch
import io
import sys
from unittest.mock import patch
from src.device_utils import (
    get_device,
    set_seed,
    print_device_info,
)


class TestDeviceUtils(unittest.TestCase):
    """Test cases for device utilities module."""

    def test_get_device(self):
        """Test getting the appropriate device (CPU or CUDA) based on availability."""
        # This test assumes that the system conditions won't change during the test
        device, message = get_device()

        # Check that the returned device is a torch.device object
        self.assertIsInstance(device, torch.device)

        # Check that the message contains either "CPU" or "GPU"
        self.assertTrue("CPU" in message or "GPU" in message)

        # Verify that the device matches the message
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
            self.assertIn("CUDA available", message)
            self.assertIn("GPU", message)
        else:
            self.assertEqual(device.type, "cpu")
            self.assertIn("CPU", message)

    def test_set_seed(self):
        """Test setting random seed for reproducibility."""
        # Test with specific seed
        test_seed = 42
        returned_seed = set_seed(test_seed)

        # The returned seed should match what we set
        self.assertEqual(returned_seed, test_seed)

        # Create two random tensors with the same seed
        torch.manual_seed(test_seed)
        tensor1 = torch.rand(5)

        torch.manual_seed(test_seed)
        tensor2 = torch.rand(5)

        # They should be identical
        self.assertTrue(torch.equal(tensor1, tensor2))

        # Test without specified seed (should return a seed, but we don't know what)
        returned_seed = set_seed()
        self.assertIsInstance(returned_seed, int)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_device_info(self, mock_stdout):
        """Test printing device information."""
        # Call the function
        info = print_device_info()

        # Check that it printed something
        output = mock_stdout.getvalue()
        self.assertGreater(len(output), 0)

        # Check that the output mentions device and seed
        self.assertIn("seed", output.lower())
        self.assertTrue("CPU" in output or "GPU" in output)

        # Check that the returned info is a dictionary with the right keys
        self.assertIsInstance(info, dict)
        self.assertIn('device', info)
        self.assertIn('seed', info)
        self.assertIn('cuda_available', info)

        # Check that the cuda_available flag matches the device
        if info['device'].type == 'cuda':
            self.assertTrue(info['cuda_available'])
        else:
            self.assertFalse(info['cuda_available'])


if __name__ == "__main__":
    unittest.main()
