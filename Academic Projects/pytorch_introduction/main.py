"""
PyTorch Introduction - Main Demo Script

This script demonstrates the usage of all modules in the PyTorch Introduction project.
"""

import torch
from src import device_utils, tensor_operations, image_processing, tensor_statistics


def demonstrate_device_setup():
    """Demonstrate GPU/CPU device setup and seed initialization."""
    print("\n" + "=" * 50)
    print("DEVICE SETUP DEMONSTRATION")
    print("=" * 50)

    # Initialize device and print info
    device_info = device_utils.print_device_info()

    # Set a specific seed and print the result
    print(f"\nSetting seed to 1234")
    seed = device_utils.set_seed(1234)
    print(f"Seed set to: {seed}")

    return device_info["device"]


def demonstrate_basic_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("\n" + "=" * 50)
    print("BASIC TENSOR OPERATIONS DEMONSTRATION")
    print("=" * 50)

    # Create random tensors
    print("\n1. Creating random tensors:")
    tensor_7x7 = tensor_operations.create_random_tensor((7, 7), seed=0)
    print(f"Random tensor with shape {tensor_7x7.shape}:\n{tensor_7x7}")

    tensor_1x7 = tensor_operations.create_random_tensor((1, 7), seed=0)
    print(
        f"\nAnother random tensor with shape {tensor_1x7.shape}:\n{tensor_1x7}")

    # Perform matrix multiplication
    result_matmul = tensor_operations.matrix_multiplication(
        tensor_7x7, tensor_1x7.t())
    print(
        f"\n2. Matrix multiplication result shape {result_matmul.shape}:\n{result_matmul}")

    # Perform element-wise multiplication
    result_elemul = tensor_operations.element_wise_multiplication(
        tensor_7x7, tensor_1x7.t())
    print(
        f"\n3. Element-wise multiplication result shape {result_elemul.shape}:\n{result_elemul}")

    # Find min and max values
    min_val, max_val, min_idx, max_idx = tensor_operations.find_min_max(
        result_matmul)
    print(f"\n4. Min value: {min_val}, at index: {min_idx}")
    print(f"   Max value: {max_val}, at index: {max_idx}")

    # Squeeze and unsqueeze tensors
    tensor_1110 = tensor_operations.create_random_tensor((1, 1, 1, 10), seed=7)
    print(f"\n5. Tensor with shape {tensor_1110.shape}:\n{tensor_1110}")

    squeezed_tensor = tensor_operations.squeeze_tensor(tensor_1110)
    print(f"   After squeezing, shape: {squeezed_tensor.shape}")
    print(f"   Squeezed tensor: {squeezed_tensor}")

    unsqueezed_tensor = tensor_operations.unsqueeze_tensor(
        squeezed_tensor, dim=0)
    print(f"   After unsqueezing at dim 0, shape: {unsqueezed_tensor.shape}")

    return tensor_7x7, tensor_1x7


def demonstrate_image_processing():
    """Demonstrate image processing with tensors."""
    print("\n" + "=" * 50)
    print("IMAGE PROCESSING DEMONSTRATION")
    print("=" * 50)

    # Create synthetic images
    print("\n1. Creating synthetic images:")
    image1 = image_processing.create_synthetic_image(7, 7, seed=42)
    print(f"Single synthetic image with shape {image1.shape}:\n{image1}")

    # Create batch of images
    batch_size = 3
    images = image_processing.create_image_batch(
        batch_size, 7, 7, channels=1, seed=42)
    print(f"\n2. Batch of {batch_size} images with shape {images.shape}")
    print(f"First image in batch:\n{images[0, 0]}")
    print(f"Third image in batch:\n{images[2, 0]}")

    # Extract rows and columns
    odd_rows = image_processing.get_image_rows(images[0], even_rows=False)
    print(f"\n3. Odd rows from first image:\n{odd_rows}")

    even_rows = image_processing.get_image_rows(images[0], even_rows=True)
    print(f"\n4. Even rows from first image:\n{even_rows}")

    odd_columns = image_processing.get_image_columns(
        images[0], even_columns=False)
    print(f"\n5. Odd columns from first image:\n{odd_columns}")

    even_columns = image_processing.get_image_columns(
        images[2], even_columns=True)
    print(f"\n6. Even columns from third image:\n{even_columns}")

    # Flatten images
    flattened_images = image_processing.flatten_image(images)
    print(f"\n7. Flattened images shape: {flattened_images.shape}")
    print(f"   Flattened first image: {flattened_images[0, 0]}")

    return images


def demonstrate_tensor_statistics():
    """Demonstrate statistical operations on tensors."""
    print("\n" + "=" * 50)
    print("TENSOR STATISTICS DEMONSTRATION")
    print("=" * 50)

    # Create a new batch of synthetic images for demonstration
    images = image_processing.create_image_batch(3, 7, 7, channels=1, seed=42)

    # Calculate basic statistics
    print("\n1. Basic tensor statistics:")
    stats = tensor_statistics.tensor_statistics(images)
    for stat_name, stat_value in stats.items():
        print(f"   {stat_name}: {stat_value}")

    # Calculate row-wise and column-wise sums
    print("\n2. Row-wise sums for first image:")
    row_sums = tensor_statistics.row_wise_sum(images[0, 0])
    print(row_sums)

    print("\n3. Column-wise sums for first image:")
    col_sums = tensor_statistics.column_wise_sum(images[0, 0])
    print(col_sums)

    # Calculate image batch statistics
    print("\n4. Image batch statistics:")
    image_stats, batch_stats = tensor_statistics.image_batch_statistics(images)

    print("\n   Individual image statistics:")
    for img_stat in image_stats:
        print(f"   Image {img_stat['image_id']}:")
        print(f"     Mean: {img_stat['mean']:.2f}")
        print(f"     Std: {img_stat['std']:.2f}")

    print("\n   Batch statistics:")
    for stat_name, stat_value in batch_stats.items():
        print(f"     {stat_name}: {stat_value}")

    # Find mode
    mode_val, mode_freq = tensor_statistics.find_mode(images)
    print(f"\n5. Mode value: {mode_val}, frequency: {mode_freq}")

    return stats


def main():
    """Main function to demonstrate all features."""
    print("\n" + "#" * 60)
    print("# PYTORCH INTRODUCTION - DEMONSTRATION OF ALL FUNCTIONALITY #")
    print("#" * 60)

    # Demonstrate device setup
    device = demonstrate_device_setup()

    # Demonstrate basic tensor operations
    tensors = demonstrate_basic_tensor_operations()

    # Demonstrate image processing
    images = demonstrate_image_processing()

    # Demonstrate tensor statistics
    stats = demonstrate_tensor_statistics()

    print("\n" + "#" * 60)
    print("# DEMONSTRATION COMPLETE #")
    print("#" * 60)


if __name__ == "__main__":
    main()
